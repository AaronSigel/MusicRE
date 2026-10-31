import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from test_shared_pipeline import pipeline_env


@pytest.fixture
def tonal_pipeline_env(pipeline_env, monkeypatch):
    from music_re.core.backends.transcriber_base import save_transcription

    pipeline, ws, source, counts = pipeline_env
    separate = pipeline.separate_stems
    get_backend = pipeline.get_transcriber_backend

    def tonal_separate(*args, **kwargs):
        result = separate(*args, **kwargs)
        sr = 44100
        tone = .5 * np.sin(2 * np.pi * 110 * np.arange(sr * 2) / sr)
        tone[sr:] = 0
        sf.write(result.stems['bass'], tone, sr)
        return result

    def backend(**kwargs):
        original = get_backend(**kwargs)

        def transcribe(audio_path, output_midi_path, **options):
            if options['stem_name'] != 'bass':
                return original.transcribe(audio_path, output_midi_path, **options)
            counts['transcription'] += 1
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            instrument.notes = [pretty_midi.Note(velocity=100, pitch=45, start=.1, end=.75),
                                pretty_midi.Note(velocity=100, pitch=80, start=1.25, end=1.75)]
            midi.instruments.append(instrument)
            return save_transcription(midi, output_midi_path, midi if options['save_raw'] else None, 'fake', {})

        return SimpleNamespace(transcribe=transcribe)

    monkeypatch.setattr(pipeline, 'separate_stems', tonal_separate)
    monkeypatch.setattr(pipeline, 'get_transcriber_backend', backend)
    return pipeline_env


@pytest.fixture
def stage_calls(pipeline_env, monkeypatch):
    pipeline = pipeline_env[0]
    calls = {'analysis': 0, 'evaluation': 0}
    for attribute, key in [('analyze_audio', 'analysis'), ('evaluate_quality', 'evaluation')]:
        original = getattr(pipeline, attribute)

        def spy(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(pipeline, attribute, spy)
    return calls


def test_off_keeps_original_artifacts(pipeline_env):
    pipeline, ws, source, _ = pipeline_env
    result = pipeline.run_pipeline(source, workspace=ws, profile='fast', spectral_validation='off')
    manifest = ws.get_manifest(source)
    assert result['active_midis'] == manifest.midis
    assert manifest.active_midi_variant == 'original'
    assert manifest.validated_midis == {}
    assert manifest.spectral_validation is None
    assert not (ws.get_track_dir(source) / 'midis_validated').exists()
    assert not (ws.get_track_dir(source) / 'spectral_validation.json').exists()


def test_invalid_mode_does_not_write_manifest(pipeline_env):
    pipeline, ws, source, _ = pipeline_env
    with pytest.raises(ValueError):
        pipeline.run_pipeline(source, workspace=ws, spectral_validation='invalid')
    assert list(ws.cache_root.iterdir()) == []


def test_modes_and_cache_isolation(pipeline_env, stage_calls):
    pipeline, ws, source, counts = pipeline_env
    initial = pipeline.run_pipeline(source, workspace=ws, spectral_validation='off')
    original = {name: Path(path).read_bytes() for name, path in initial['midis'].items()}
    upstream = {name: ws.get_manifest(source).stage_fingerprints[name] for name in pipeline.STAGES[:4]}
    for mode in ('audit', 'filter', 'off'):
        result = pipeline.run_pipeline(source, workspace=ws, spectral_validation=mode)
        manifest = ws.get_manifest(source)
        assert {name: Path(path).read_bytes() for name, path in manifest.midis.items()} == original
        assert {name: manifest.stage_fingerprints[name] for name in pipeline.STAGES[:4]} == upstream
        assert counts == dict(separation=1, structure=1, transcription=4)
        assert result['active_midis'] == (manifest.validated_midis if mode == 'filter' else manifest.midis)
        assert (ws.get_track_dir(source) / 'spectral_validation.json').exists() == (mode != 'off')
        if mode == 'filter':
            assert set(manifest.validated_midis) == set(manifest.midis)
            assert all(Path(path).is_file() for path in manifest.validated_midis.values())
    assert stage_calls == {'analysis': 2, 'evaluation': 4}


def test_corrupt_report_reruns_stage(pipeline_env, monkeypatch):
    pipeline, ws, source, counts = pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='audit')
    report = ws.get_track_dir(source) / 'spectral_validation.json'
    report.write_text('corrupt')
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='audit')
    assert report.read_text() != 'corrupt'
    assert counts == dict(separation=1, structure=1, transcription=4)


def test_validator_error_is_recorded(pipeline_env, monkeypatch):
    pipeline, ws, source, _ = pipeline_env
    def fail(*args, **kwargs):
        raise RuntimeError('validator failed')
    monkeypatch.setattr(pipeline.SpectralMidiValidator, 'validate', fail)
    with pytest.raises(RuntimeError, match='validator failed'):
        pipeline.run_pipeline(source, workspace=ws, spectral_validation='filter')
    manifest = ws.get_manifest(source)
    assert manifest.status == 'failed'
    assert manifest.stage_errors['spectral_validation'] == 'validator failed'
    assert manifest.active_midi_variant == 'original'


def test_filter_export_includes_both_midi_sets_and_report(tonal_pipeline_env, tmp_path):
    from music_re.core.export import export_bundle, validate_bundle
    pipeline, ws, source, _ = tonal_pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='filter')
    manifest = ws.get_manifest(source)
    validate_bundle(ws, source)
    target = export_bundle(ws, source, tmp_path / 'bundle')
    for name in manifest.midis:
        assert (target / 'midis' / f'{name}.mid').read_bytes() == Path(manifest.validated_midis[name]).read_bytes()
        assert (target / 'midis_original' / f'{name}.mid').read_bytes() == Path(manifest.midis[name]).read_bytes()
    assert (target / 'spectral_validation.json').is_file()
    original = pretty_midi.PrettyMIDI(str(target / 'midis_original' / 'bass.mid'))
    filtered = pretty_midi.PrettyMIDI(str(target / 'midis' / 'bass.mid'))
    original_notes = [note for instrument in original.instruments for note in instrument.notes]
    filtered_notes = [note for instrument in filtered.instruments for note in instrument.notes]
    assert [note.pitch for note in original_notes] == [45, 80]
    assert [note.pitch for note in filtered_notes] == [45]
    notes = json.loads((target / 'spectral_validation.json').read_text())['stems']['bass']['notes']
    assert len(filtered_notes) == sum(note['decision'] != 'reject' for note in notes)


def test_config_change_only_reruns_spectral_and_evaluation(pipeline_env, monkeypatch, stage_calls):
    pipeline, ws, source, counts = pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='audit')
    old = ws.get_manifest(source).stage_fingerprints.copy()
    config = pipeline.SpectralValidatorConfig()
    changed = replace(config, spectral=replace(config.spectral, hop_length=config.spectral.hop_length * 2))
    monkeypatch.setattr(pipeline, 'SpectralValidatorConfig', lambda: changed)
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='audit')
    new = ws.get_manifest(source).stage_fingerprints
    assert all(old[stage] == new[stage] for stage in pipeline.STAGES[:4])
    assert all(old[stage] != new[stage] for stage in pipeline.STAGES[4:])
    assert counts == dict(separation=1, structure=1, transcription=4)
    assert stage_calls == {'analysis': 2, 'evaluation': 2}


def test_legacy_manifest_reuses_upstream_fingerprints(pipeline_env, stage_calls):
    import json
    pipeline, ws, source, counts = pipeline_env
    pipeline.run_pipeline(source, workspace=ws)
    manifest_path = ws.get_manifest_path(source)
    raw = json.loads(manifest_path.read_text())
    previous = pipeline.compute_file_content_hash(source)
    for stage in pipeline.STAGES[:4]:
        previous = pipeline._fingerprint([stage, previous, raw['run_config'], raw['runtime']['packages'], raw['runtime']['weights'], 'corrective-v3'])
        assert raw['stage_fingerprints'][stage] == previous
    for key in ('validated_midis', 'active_midi_variant', 'spectral_validation'):
        raw.pop(key)
    raw['completed_stages'].remove('spectral_validation')
    raw['stage_fingerprints'].pop('spectral_validation')
    raw['artifact_hashes'].pop('spectral_validation')
    raw['stage_fingerprints']['evaluation'] = pipeline._fingerprint(['evaluation', previous, raw['run_config'], raw['runtime']['packages'], raw['runtime']['weights'], 'corrective-v3'])
    manifest_path.write_text(json.dumps(raw))
    pipeline.run_pipeline(source, workspace=ws)
    assert counts == dict(separation=1, structure=1, transcription=4)
    assert stage_calls == {'analysis': 2, 'evaluation': 2}
    assert 'spectral_validation' in ws.get_manifest(source).completed_stages


def test_failed_corrupt_report_revalidation_is_not_completed(pipeline_env, monkeypatch):
    pipeline, ws, source, _ = pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='audit')
    ws.get_spectral_validation_path(source).write_text('corrupt')
    def fail(*args, **kwargs):
        raise RuntimeError('validator failed')
    monkeypatch.setattr(pipeline.SpectralMidiValidator, 'validate', fail)
    with pytest.raises(RuntimeError):
        pipeline.run_pipeline(source, workspace=ws, spectral_validation='audit')
    assert 'spectral_validation' not in ws.get_manifest(source).completed_stages


def test_audit_report_contains_unscored_original_notes(tonal_pipeline_env):
    from music_re.schemas import SpectralValidationResult
    pipeline, ws, source, _ = tonal_pipeline_env
    result = pipeline.run_pipeline(source, workspace=ws, spectral_validation='audit')
    report = SpectralValidationResult.model_validate_json(ws.get_spectral_validation_path(source).read_text())
    assert report.mode == 'audit'
    assert report.active_variant == 'original'
    assert result['active_midis'] == result['midis']
    assert len(report.stems['bass'].notes) == 2
    assert all(note.decision == 'unscored' for stem in report.stems.values() for note in stem.notes)


def test_quality_uses_filtered_midis(tonal_pipeline_env, monkeypatch):
    import music_re.core.quality as quality
    pipeline, ws, source, _ = tonal_pipeline_env
    actual = quality.transcription_quality
    calls = []

    def spy(midis, ground_truth):
        calls.append(dict(midis))
        return actual(midis, ground_truth)

    monkeypatch.setattr(quality, 'transcription_quality', spy)
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='filter')
    manifest = ws.get_manifest(source)
    assert calls == [manifest.validated_midis]
    assert manifest.evaluation['report']['transcription_metrics']['total_notes'] == 1


def test_filter_dawproject_contains_validated_notes(tonal_pipeline_env, tmp_path):
    from dawproject import DawProject
    from music_re.core.export import export_bundle
    pipeline, ws, source, _ = tonal_pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='filter')
    target = export_bundle(ws, source, tmp_path / 'bundle', export_dawproject=True)
    project = DawProject.load_project(str(target / 'song.dawproject'))
    bass_lanes = [lane for lane in project.arrangement.lanes.lanes if lane.track.name == 'bass (MIDI)']
    assert len(bass_lanes) == 1
    assert [note.key for clip in bass_lanes[0].clips for note in clip.content.notes] == [45]


def test_off_export_has_original_layout(pipeline_env, tmp_path):
    from music_re.core.export import export_bundle
    pipeline, ws, source, _ = pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='off')
    target = export_bundle(ws, source, tmp_path / 'bundle')
    files = {str(path.relative_to(target)) for path in target.rglob('*') if path.is_file()}
    names = {'vocals', 'drums', 'bass', 'other'}
    expected = {'manifest.json', 'timeline.json', 'metadata.json', 'quality_report.json', 'midis/all_stems.mid'}
    expected |= {f'stems/{name}.wav' for name in names}
    expected |= {f'midis/{name}.mid' for name in names}
    expected |= {f'midis/{name}_raw.mid' for name in names}
    assert files == expected


def test_manifest_and_result_hold_summary_without_notes(tonal_pipeline_env):
    import json
    from music_re.schemas import SpectralValidationResult
    pipeline, ws, source, _ = tonal_pipeline_env
    result = pipeline.run_pipeline(source, workspace=ws, spectral_validation='filter')
    manifest = ws.get_manifest(source)
    report_path = ws.get_spectral_validation_path(source)
    summary = manifest.spectral_validation
    assert result['spectral_validation'] == summary
    assert set(summary) == {'mode', 'algo_version', 'active_variant', 'report_path', 'stems'}
    assert summary['mode'] == 'filter'
    assert summary['active_variant'] == 'validated'
    assert summary['report_path'] == str(report_path)
    assert '"notes":' not in json.dumps(summary)
    bass = summary['stems']['bass']
    assert set(bass) == {'status', 'counts', 'warnings', 'input_midi_path', 'output_midi_path'}
    assert bass['input_midi_path'] == manifest.midis['bass']
    assert bass['output_midi_path'] == manifest.validated_midis['bass']
    report = SpectralValidationResult.model_validate_json(report_path.read_text())
    assert len(report.stems['bass'].notes) == 2
    assert bass['counts'] == report.stems['bass'].counts
    assert '"notes":' not in ws.get_manifest_path(source).read_text()


def test_legacy_manifest_with_full_report_loads_and_exports(tonal_pipeline_env, tmp_path):
    import json
    from music_re.core.export import export_bundle, validate_bundle
    pipeline, ws, source, _ = tonal_pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='filter')
    manifest_path = ws.get_manifest_path(source)
    raw = json.loads(manifest_path.read_text())
    raw['spectral_validation'] = json.loads(ws.get_spectral_validation_path(source).read_text())
    manifest_path.write_text(json.dumps(raw))
    assert ws.get_manifest(source).spectral_validation['stems']['bass']['notes']
    validate_bundle(ws, source)
    export_bundle(ws, source, tmp_path / 'bundle')


def test_export_rejects_validated_midi_disagreeing_with_counts(tonal_pipeline_env):
    from music_re.core.export import validate_bundle
    pipeline, ws, source, _ = tonal_pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='filter')
    manifest = ws.get_manifest(source)
    manifest.spectral_validation['stems']['bass']['counts'] = {'keep': 2}
    ws.save_manifest(source, manifest)
    with pytest.raises(ValueError, match='note count'):
        validate_bundle(ws, source)


@pytest.mark.parametrize('mode', ['AUDIT', ' audit '])
def test_run_pipeline_normalises_mode(pipeline_env, mode):
    pipeline, ws, source, _ = pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation=mode)
    assert ws.get_manifest(source).spectral_validation['mode'] == 'audit'


def test_stem_report_has_no_failed_status_or_error_field():
    from pydantic import ValidationError
    from music_re.schemas import StemValidationReport
    base = dict(stem='bass', mode='audit', input_midi_path='bass.mid')
    assert 'error' not in StemValidationReport.model_fields
    with pytest.raises(ValidationError):
        StemValidationReport(status='failed', **base)
    legacy = StemValidationReport.model_validate(dict(base, status='validated', error=None))
    assert legacy.status == 'validated'
