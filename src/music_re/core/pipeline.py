"""One stage/state owner for CLI and MCP."""
import hashlib
import json
import random
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from music_re.core.analyzer import analyze_audio
from music_re.core.audio_io import decode_audio
from music_re.core.backends import get_structure_analyzer, get_transcriber_backend
from music_re.core.backends.separator_base import resolve_device
from music_re.core.chords import detect_chords_for_bars
from music_re.core.midi_project import build_combined_midi_project
from music_re.core.midi_validation import SpectralMidiValidator, SpectralValidatorConfig, SPECTRAL_ALGO_VERSION
from music_re.core.profiles import resolve_profile
from music_re.core.provenance import model_inventory
from music_re.core.quality import evaluate_quality
from music_re.core.separator import separate_stems
from music_re.schemas import BarChord, SpectralValidationResult
from music_re.workspace import WorkspaceManager, compute_file_content_hash, active_midis, _atomic_write_json

STAGES = ['separation', 'analysis', 'structure', 'transcription', 'spectral_validation', 'evaluation']


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _hashes(paths):
    return {str(path): compute_file_content_hash(path) for path in paths}


def _fingerprints(source, config, packages, weights, mode, validator_config):
    previous = compute_file_content_hash(source)
    fingerprints = {}
    for stage in STAGES:
        if stage == 'spectral_validation':
            value = [stage, previous, mode, validator_config.to_dict(), SPECTRAL_ALGO_VERSION]
        else:
            value = [stage, previous, config, packages, weights, 'corrective-v3']
        previous = fingerprints[stage] = _fingerprint(value)
    return fingerprints


def _valid(paths, hashes):
    try:
        return bool(paths) and all(Path(p).is_file() and hashes.get(str(p)) == compute_file_content_hash(p) for p in paths)
    except OSError:
        return False


def _validation_summary(result, report_path):
    """Per-note evidence stays in the report file; the manifest keeps only this summary."""
    stems = {name: {key: stem[key] for key in ('status', 'counts', 'warnings', 'input_midi_path', 'output_midi_path')}
             for name, stem in result['stems'].items()}
    return dict(mode=result['mode'], algo_version=result['algo_version'], active_variant=result['active_variant'],
                report_path=str(report_path), stems=stems)


def run_pipeline(file_path, workspace=None, six_stems=False, device='auto', profile='balanced',
                 progress_callback=None, separator_backend='demucs', spectral_validation: str = 'off') -> dict:
    spectral_validation = str(spectral_validation).strip().lower()
    if spectral_validation not in ('off', 'audit', 'filter'):
        raise ValueError('invalid spectral_validation mode')
    settings = resolve_profile(profile, six_stems)
    resolved_device = resolve_device(device)
    source = Path(file_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    ws = workspace if workspace is not None else WorkspaceManager()
    
    with ws.lock(source):
        config = settings.to_dict() | dict(requested_device=device, device=resolved_device, separator_backend=separator_backend)
        packages = {name: version(name) for name in ('music-re', 'torch', 'torchaudio', 'demucs',
                                                    'all-in-one-infer', 'basic-pitch', 'librosa', 'mir-eval', 'museval')}
        separator_model_id = 'scnet' if separator_backend == 'scnet' else ('htdemucs_6s' if settings.six_stems else 'htdemucs')
        runtime = dict(packages=packages, python=__import__('platform').python_version(),
                       model_ids=dict(separator=separator_model_id,
                                      structure=settings.structure_model, tonal='icassp_2022/nmp.onnx', drums='multiband'))
        runtime['weights'] = model_inventory(runtime['model_ids']['separator'], settings.structure_model)
        root = ws.get_track_dir(source)
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        ws.update_manifest(source, status='processing', run_config=config, runtime=runtime)
        # Fingerprints include effective settings, library versions and upstream stages.
        validator_config = SpectralValidatorConfig()
        fingerprints = _fingerprints(source, config, packages, runtime['weights'], spectral_validation, validator_config)

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
                    'spectral_validation': ([] if spectral_validation == 'off' else
                                            [ws.get_spectral_validation_path(source)] + list(manifest.validated_midis.values())),
                    'evaluation': [ws.get_quality_report_path(source)],
                }[stage]
                if not changed and stage in manifest.completed_stages and ((stage == 'spectral_validation' and spectral_validation == 'off') or _valid(paths, manifest.artifact_hashes.get(stage, {}))):
                    if progress_callback:
                        progress_callback(f'Reusing {stage}.')
                    continue
                # Upstream files can change without changing their paths.
                ws.invalidate_stages(source, STAGES[STAGES.index(stage) + 1:])
                if progress_callback:
                    progress_callback(f'Running {stage}.')
                if stage == 'separation':
                    result = separate_stems(source, workspace=ws, six_stems=settings.six_stems,
                                            backend_name=separator_backend,
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

                    bars = []
                    if timeline.downbeats and len(timeline.downbeats) >= 2:
                        for i in range(len(timeline.downbeats) - 1):
                            bars.append((timeline.downbeats[i], timeline.downbeats[i + 1]))
                        if timeline.downbeats[-1] < timeline.duration_sec:
                            bars.append((timeline.downbeats[-1], timeline.duration_sec))
                    elif timeline.beats and len(timeline.beats) >= 2:
                        bar_starts = timeline.beats[::4]
                        for i in range(len(bar_starts) - 1):
                            bars.append((bar_starts[i], bar_starts[i + 1]))
                        if bar_starts and bar_starts[-1] < timeline.duration_sec:
                            bars.append((bar_starts[-1], timeline.duration_sec))

                    if bars:
                        sr = 22050
                        y = None
                        if manifest.stems and any(s in manifest.stems for s in ('piano', 'guitar', 'other', 'bass')):
                            stem_signals = []
                            for s_name in ('piano', 'guitar', 'other', 'bass'):
                                if s_name in manifest.stems and Path(manifest.stems[s_name]).is_file():
                                    try:
                                        sig, _ = decode_audio(manifest.stems[s_name], target_sr=sr, normalize_db=None)
                                        if sig.ndim > 1:
                                            sig = np.mean(sig, axis=1)
                                        stem_signals.append(sig)
                                    except Exception:
                                        pass
                            if stem_signals:
                                max_len = max(len(s) for s in stem_signals)
                                y = np.zeros(max_len, dtype=np.float32)
                                for s in stem_signals:
                                    y[:len(s)] += s

                        if y is None:
                            try:
                                y, _ = decode_audio(source, target_sr=sr, normalize_db=None)
                                if y.ndim > 1:
                                    y = np.mean(y, axis=1)
                            except Exception:
                                y = None

                        if y is not None and len(y) > 0:
                            detected_chords = detect_chords_for_bars(y, sr, bars)
                            timeline.chords = [
                                BarChord(**c) if isinstance(c, dict) else c for c in detected_chords
                            ]

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
                        if name == 'drums':
                            try:
                                transcriber = get_transcriber_backend(mode='drums', name='adtof')
                            except Exception:
                                transcriber = get_transcriber_backend(mode='drums')
                        elif name == 'piano':
                            try:
                                transcriber = get_transcriber_backend(mode='tonal', name='bytedance_piano')
                            except Exception:
                                transcriber = get_transcriber_backend(mode='tonal', name='basic_pitch')
                        elif name == 'vocals':
                            try:
                                transcriber = get_transcriber_backend(mode='tonal', name='crepe_vocal')
                            except Exception:
                                transcriber = get_transcriber_backend(mode='tonal', name='basic_pitch')
                        elif name in ('bass', 'guitar', 'other'):
                            transcriber = get_transcriber_backend(mode='tonal', name='basic_pitch')
                        else:
                            transcriber = get_transcriber_backend(mode='tonal')
                        result = transcriber.transcribe(stem, midi_dir / f'{name}.mid', timeline=timeline, bpm=bpm,
                                                         save_raw=settings.save_raw, grid_subdivisions=settings.grid_subdivisions,
                                                         quantize_ends=settings.quantize_ends, stem_name=name)
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
                elif stage == 'spectral_validation':
                    ws.invalidate_stages(source, [stage])
                    report_path = ws.get_spectral_validation_path(source)
                    midi_dir = ws.get_validated_midis_dir(source)
                    if spectral_validation == 'off':
                        if midi_dir.exists():
                            shutil.rmtree(midi_dir)
                        report_path.unlink(missing_ok=True)
                        paths = []
                        ws.mark_stage_complete(source, stage)
                    else:
                        if spectral_validation == 'filter':
                            midi_dir.mkdir(parents=True, exist_ok=True)
                        validator = SpectralMidiValidator(validator_config)
                        reports, validated = {}, {}
                        for name, midi in manifest.midis.items():
                            if name == 'all_stems':
                                continue
                            output = midi_dir / f'{name}.mid' if spectral_validation == 'filter' else None
                            reports[name] = validator.validate(manifest.stems[name], midi, name, spectral_validation,
                                                             out_midi_path=output, mix_audio_path=source)
                            if output is not None:
                                validated[name] = str(output)
                        if spectral_validation == 'filter':
                            timeline = ws.load_timeline(source)
                            bpm = timeline.bpm if timeline and timeline.bpm > 0 else (manifest.analysis or {}).get('bpm', 120)
                            if bpm <= 0:
                                bpm = 120.0
                            validated['all_stems'] = str(build_combined_midi_project(validated, midi_dir / 'all_stems.mid', timeline=timeline, bpm=bpm))
                        result = SpectralValidationResult(mode=spectral_validation, algo_version=SPECTRAL_ALGO_VERSION,
                                                          config=validator_config.to_dict(), stems=reports,
                                                          active_variant='validated' if validated else 'original').model_dump()
                        _atomic_write_json(report_path, result)
                        ws.update_manifest(source, validated_midis=validated, active_midi_variant=result['active_variant'],
                                           spectral_validation=_validation_summary(result, report_path))
                        ws.mark_stage_complete(source, stage)
                        paths = [report_path] + list(validated.values())
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
            final_fingerprints = _fingerprints(source, config, packages, runtime['weights'], spectral_validation, validator_config)
            for completed_stage in STAGES:
                if completed_stage in m.completed_stages:
                    m.stage_fingerprints[completed_stage] = final_fingerprints[completed_stage]
            m.status, m.current_stage, m.runtime = 'completed', 'completed', runtime
            ws.save_manifest(source, m)
        except Exception as e:
            ws.record_stage_error(source, stage, str(e))
            raise
        manifest = ws.get_manifest(source)
        return dict(workspace_dir=str(root), analysis=manifest.analysis, timeline=manifest.timeline,
                    stems=manifest.stems, midis=manifest.midis, active_midis=active_midis(manifest),
                    spectral_validation=manifest.spectral_validation,
                    quality_report=manifest.evaluation.get('report') if manifest.evaluation else None,
                    transcription_results=manifest.transcription_results, provenance=manifest.runtime)
