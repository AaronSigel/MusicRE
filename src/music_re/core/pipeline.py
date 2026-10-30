"""One stage/state owner for CLI and MCP."""
import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from music_re.core.analyzer import analyze_audio
from music_re.core.backends import get_structure_analyzer, get_transcriber_backend
from music_re.core.backends.separator_base import resolve_device
from music_re.core.midi_project import build_combined_midi_project
from music_re.core.profiles import resolve_profile
from music_re.core.provenance import model_inventory
from music_re.core.quality import evaluate_quality
from music_re.core.separator import separate_stems
from music_re.workspace import WorkspaceManager, compute_file_content_hash

STAGES = ['separation', 'analysis', 'structure', 'transcription', 'evaluation']


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _hashes(paths):
    return {str(path): compute_file_content_hash(path) for path in paths}


def _valid(paths, hashes):
    try:
        return bool(paths) and all(Path(p).is_file() and hashes.get(str(p)) == compute_file_content_hash(p) for p in paths)
    except OSError:
        return False


def run_pipeline(file_path, workspace=None, six_stems=False, device='auto', profile='balanced',
                 progress_callback=None) -> dict:
    settings = resolve_profile(profile, six_stems)
    resolved_device = resolve_device(device)
    source = Path(file_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    ws = workspace if workspace is not None else WorkspaceManager()
    
    with ws.lock(source):
        config = settings.to_dict() | dict(requested_device=device, device=resolved_device)
        packages = {name: version(name) for name in ('music-re', 'torch', 'torchaudio', 'demucs',
                                                    'all-in-one-infer', 'basic-pitch', 'librosa', 'mir-eval', 'museval')}
        runtime = dict(packages=packages, python=__import__('platform').python_version(),
                       model_ids=dict(separator='htdemucs_6s' if settings.six_stems else 'htdemucs',
                                      structure=settings.structure_model, tonal='icassp_2022/nmp.onnx', drums='multiband'))
        runtime['weights'] = model_inventory(runtime['model_ids']['separator'], settings.structure_model)
        root = ws.get_track_dir(source)
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        ws.update_manifest(source, status='processing', run_config=config, runtime=runtime)
        # Fingerprints include effective settings, library versions and upstream stages.
        previous = compute_file_content_hash(source)
        fingerprints = {}
        for stage in STAGES:
            fingerprints[stage] = _fingerprint([stage, previous, config, packages, runtime['weights'], 'corrective-v3'])
            previous = fingerprints[stage]

        stage = 'separation'
        try:
            for stage in STAGES:
                if stage == 'evaluation' and not settings.evaluate:
                    ws.invalidate_stages(source, ['evaluation'])
                    break
                manifest = ws.get_manifest(source)
                current = fingerprints[stage]
                changed = manifest.stage_fingerprints.get(stage) != current
                if changed:
                    ws.invalidate_stages(source, STAGES[STAGES.index(stage):])
                    manifest = ws.get_manifest(source)
                ws.update_manifest(source, current_stage=stage, status='processing')
                paths = {
                    'separation': list(manifest.stems.values()),
                    'analysis': [root / 'metadata.json'],
                    'structure': [ws.get_timeline_path(source)],
                    'transcription': list(manifest.midis.values()) + [v['raw_midi_path'] for v in manifest.transcription_results.values() if v.get('raw_midi_path')],
                    'evaluation': [ws.get_quality_report_path(source)],
                }[stage]
                if not changed and stage in manifest.completed_stages and _valid(paths, manifest.artifact_hashes.get(stage, {})):
                    if progress_callback:
                        progress_callback(f'Reusing {stage}.')
                    continue
                # Upstream files can change without changing their paths.
                ws.invalidate_stages(source, STAGES[STAGES.index(stage) + 1:])
                if progress_callback:
                    progress_callback(f'Running {stage}.')
                if stage == 'separation':
                    result = separate_stems(source, workspace=ws, six_stems=settings.six_stems,
                                            device=resolved_device, shifts=settings.shifts, overlap=settings.overlap,
                                            force=not changed,
                                            seed=settings.seed,
                                            progress_callback=progress_callback)
                    ws.mark_stage_complete(source, stage, result.stems)
                    paths = list(result.stems.values())
                elif stage == 'analysis':
                    analysis = analyze_audio(source).model_dump()
                    if 'drums' in manifest.stems:
                        drum_analysis = analyze_audio(manifest.stems['drums'], stem_name='drums')
                        if drum_analysis.bpm > 0:
                            analysis['bpm'] = drum_analysis.bpm
                    ws.save_metadata(source, analysis)
                    ws.mark_stage_complete(source, stage, analysis)
                elif stage == 'structure':
                    kwargs = dict(model_name=settings.structure_model) if settings.structure_model else {}
                    analyzer = get_structure_analyzer(name=settings.structure_backend, **kwargs)
                    timeline = analyzer.analyze(source, stems=manifest.stems, device=resolved_device)
                    ws.save_timeline(source, timeline)
                elif stage == 'transcription':
                    timeline = ws.load_timeline(source)
                    bpm = timeline.bpm if timeline and timeline.bpm > 0 else (manifest.analysis or {}).get('bpm', 120)
                    if bpm <= 0:
                        bpm = 120.0
                    midi_dir = root / 'midis'
                    midi_dir.mkdir(parents=True, exist_ok=True)
                    midis = {k: v for k, v in manifest.midis.items() if k in manifest.stems}
                    results = dict(manifest.transcription_results)
                    old_hashes = manifest.artifact_hashes.get(stage, {})
                    for name, stem in manifest.stems.items():
                        prior = results.get(name, {})
                        prior_paths = [prior.get('midi_path', '')]
                        if settings.save_raw:
                            prior_paths.append(prior.get('raw_midi_path', ''))
                        if not changed and _valid(prior_paths, old_hashes):
                            continue
                        transcriber = get_transcriber_backend(mode='drums' if name == 'drums' else 'tonal')
                        result = transcriber.transcribe(stem, midi_dir / f'{name}.mid', timeline=timeline, bpm=bpm,
                                                         save_raw=settings.save_raw, grid_subdivisions=settings.grid_subdivisions,
                                                         quantize_ends=settings.quantize_ends)
                        results[name] = result.model_dump()
                        midis[name] = result.midi_path
                        # Persist individual successes to resume after later failures.
                        m = ws.get_manifest(source)
                        m.midis, m.transcription_results = midis, results
                        m.stage_fingerprints[stage] = current
                        result_paths = [result.midi_path] + ([result.raw_midi_path] if result.raw_midi_path else [])
                        m.artifact_hashes.setdefault(stage, {}).update(_hashes(result_paths))
                        ws.save_manifest(source, m)
                    combined = build_combined_midi_project(midis, midi_dir / 'all_stems.mid', timeline=timeline, bpm=bpm)
                    midis['all_stems'] = str(combined)
                    ws.mark_stage_complete(source, stage, midis)
                    ws.update_manifest(source, transcription_results=results)
                    paths = list(midis.values()) + [v['raw_midi_path'] for v in results.values() if v.get('raw_midi_path')]
                else:
                    report = evaluate_quality(ws, source)
                    ws.save_quality_report(source, report)
                m = ws.get_manifest(source)
                m.stage_fingerprints[stage] = current
                m.artifact_hashes[stage] = _hashes(paths)
                ws.save_manifest(source, m)
            runtime['weights'] = model_inventory(runtime['model_ids']['separator'], settings.structure_model)
            # First-time downloads become known during inference. Bind every
            # completed stage to that final inventory, not the pre-download absence.
            m = ws.get_manifest(source)
            previous = compute_file_content_hash(source)
            for completed_stage in STAGES:
                fingerprint = _fingerprint([completed_stage, previous, config, packages, runtime['weights'], 'corrective-v3'])
                previous = fingerprint
                if completed_stage in m.completed_stages:
                    m.stage_fingerprints[completed_stage] = fingerprint
            m.status, m.current_stage, m.runtime = 'completed', 'completed', runtime
            ws.save_manifest(source, m)
        except Exception as e:
            ws.record_stage_error(source, stage, str(e))
            raise
        manifest = ws.get_manifest(source)
        return dict(workspace_dir=str(root), analysis=manifest.analysis, timeline=manifest.timeline,
                    stems=manifest.stems, midis=manifest.midis,
                    quality_report=manifest.evaluation.get('report') if manifest.evaluation else None,
                    transcription_results=manifest.transcription_results, provenance=manifest.runtime)
