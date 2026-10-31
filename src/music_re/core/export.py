import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pretty_midi
import soundfile as sf

from music_re.core.dawproject_export import export_dawproject as generate_dawproject
from music_re.core.profiles import resolve_profile
from music_re.schemas import MusicalTimeline, QualityReport, TranscriptionResult
from music_re.workspace import compute_file_content_hash, active_midis


def validate_bundle(workspace, file_path, profile='balanced'):
    """Return only registered deliverables, or reject the entire incomplete bundle."""
    manifest = workspace.get_manifest(file_path)
    errors = []
    settings = resolve_profile(profile, manifest.run_config.get('six_stems', False))
    if manifest.run_config.get('profile') != profile:
        errors.append('execution/export profile mismatch or missing execution profile')
    required_stages = {'separation', 'analysis', 'structure', 'transcription'}
    if manifest.spectral_validation is not None or manifest.active_midi_variant == 'validated':
        required_stages.add('spectral_validation')
    if settings.evaluate:
        required_stages.add('evaluation')
    if manifest.status != 'completed' or not required_stages.issubset(manifest.completed_stages):
        errors.append('pipeline stages not completed')
    names = {'vocals', 'bass', 'drums', 'other'} | ({'guitar', 'piano'} if settings.six_stems else set())
    midis = active_midis(manifest)
    if set(manifest.stems) != names or set(midis) != names | {'all_stems'}:
        errors.append('active stem/MIDI sets do not match model configuration')
    files = {}
    for role in names:
        files[f'stems/{role}.wav'] = manifest.stems.get(role)
        files[f'midis/{role}.mid'] = midis.get(role)
        if manifest.active_midi_variant == 'validated':
            files[f'midis_original/{role}.mid'] = manifest.midis.get(role)
        try:
            result = TranscriptionResult(**manifest.transcription_results[role])
            if result.midi_path != manifest.midis.get(role):
                raise ValueError('result MIDI path disagrees with active artifact')
            if (result.status == 'no_notes') != (result.note_count == 0):
                raise ValueError('result state and note count disagree')
            if settings.save_raw:
                files[f'midis/{role}_raw.mid'] = result.raw_midi_path
        except Exception as e:
            errors.append(f'{role}: invalid transcription result: {e}')
    root = workspace.get_track_dir(file_path)
    files.update({'midis/all_stems.mid': midis.get('all_stems'),
                  'timeline.json': workspace.get_timeline_path(file_path),
                  'metadata.json': root / 'metadata.json'})
    if manifest.active_midi_variant == 'validated':
        files['midis_original/all_stems.mid'] = manifest.midis.get('all_stems')
    if manifest.spectral_validation is not None:
        files['spectral_validation.json'] = workspace.get_spectral_validation_path(file_path)
    if settings.evaluate:
        files['quality_report.json'] = workspace.get_quality_report_path(file_path)
    for destination, source in files.items():
        try:
            if not source or not Path(source).is_file():
                raise ValueError('missing required artifact')
            path = Path(source)
            if path.suffix == '.wav':
                info = sf.info(path)
                if info.frames <= 0:
                    raise ValueError('empty audio')
            elif path.suffix == '.mid':
                pm = pretty_midi.PrettyMIDI(str(path))
                role = Path(destination).stem
                if role in manifest.transcription_results:
                    if manifest.active_midi_variant == 'validated' and destination.startswith('midis/'):
                        notes = manifest.spectral_validation['stems'][role]['notes']
                        expected = sum(note['decision'] != 'reject' for note in notes)
                    else:
                        expected = manifest.transcription_results[role]['note_count']
                    if sum(len(i.notes) for i in pm.instruments) != expected:
                        raise ValueError('MIDI note count disagrees with result')
            elif destination == 'timeline.json':
                MusicalTimeline.model_validate_json(path.read_text())
            elif destination == 'quality_report.json':
                QualityReport.model_validate_json(path.read_text())
            else:
                json.loads(path.read_text())
        except Exception as e:
            errors.append(f'{destination}: {e}')
    for hashes in manifest.artifact_hashes.values():
        for source, digest in hashes.items():
            if source in {str(p) for p in files.values() if p}:
                try:
                    if compute_file_content_hash(source) != digest:
                        errors.append(f'{source}: artifact checksum changed')
                except OSError:
                    pass  # The missing-file error above is more useful.
    if errors:
        raise ValueError('Incomplete bundle: ' + '; '.join(errors))
    return manifest, {name: Path(path) for name, path in files.items()}


def export_bundle(workspace, file_path, target_path, profile='balanced', archive=False, export_dawproject: bool = False) -> Path:
    source = Path(file_path).resolve()
    target = Path(target_path).resolve()
    root = workspace.get_track_dir(source)
    is_direct_daw = export_dawproject and target.suffix == '.dawproject'
    if is_direct_daw:
        actual = target
    else:
        actual = target if not archive or target.suffix == '.zip' else target.with_suffix('.zip')
    for path in (target, actual):
        if path == source:
            raise ValueError('Target path cannot overwrite source input file')
        if path == root or root in path.parents:
            raise ValueError('Target path cannot be inside the track cache workspace')
        if path in root.parents or path in source.parents:
            raise ValueError('Target path cannot be an ancestor of source or cache')
    if actual.exists():
        raise ValueError(f'Target already exists: {actual}')
    manifest, files = validate_bundle(workspace, source, profile)
    actual.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.music-re-export-', dir=actual.parent) as temp:
        if is_direct_daw:
            staged_daw = Path(temp) / actual.name
            generate_dawproject(workspace, source, staged_daw)
            os.replace(staged_daw, actual)
            return actual
        staged = Path(temp) / 'bundle'
        staged.mkdir()
        for relative, original in files.items():
            dest = staged / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, dest)
        if export_dawproject:
            daw_file = staged / f'{source.stem}.dawproject'
            generate_dawproject(workspace, source, daw_file)
        portable = manifest.model_dump()
        portable['runtime']['source_input_file'] = portable['input_file']
        portable['input_file'] = source.name
        portable['stems'] = {name: f'stems/{name}.wav' for name in manifest.stems}
        portable['midis'] = {name: f'midis/{name}.mid' for name in manifest.midis}
        if manifest.active_midi_variant == 'validated':
            portable['validated_midis'] = dict(portable['midis'])
        if export_dawproject:
            portable['dawproject'] = f'{source.stem}.dawproject'
        for name, result in portable['transcription_results'].items():
            result['midi_path'] = f"{'midis_original' if manifest.active_midi_variant == 'validated' else 'midis'}/{name}.mid"
            result['raw_midi_path'] = f'midis/{name}_raw.mid' if profile != 'fast' else None
        portable['artifact_hashes'] = {'bundle': {relative: compute_file_content_hash(staged / relative) for relative in files}}
        if export_dawproject:
            daw_rel = f'{source.stem}.dawproject'
            portable['artifact_hashes']['bundle'][daw_rel] = compute_file_content_hash(staged / daw_rel)
        (staged / 'manifest.json').write_text(json.dumps(portable, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
        if archive:
            staged_zip = Path(temp) / 'bundle.zip'
            with zipfile.ZipFile(staged_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file in staged.rglob('*'):
                    if file.is_file():
                        zf.write(file, file.relative_to(staged))
            # Exclusive publication refuses overwriting a concurrent export.
            os.link(staged_zip, actual)
        else:
            staged.rename(actual)
    return actual


export_dawproject = generate_dawproject
