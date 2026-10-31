from pathlib import Path
from dataclasses import replace

import pytest

from test_shared_pipeline import pipeline_env


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


def test_modes_and_cache_isolation(pipeline_env, monkeypatch):
    pipeline, ws, source, counts = pipeline_env
    original_evaluate = pipeline.evaluate_quality
    evaluations = []
    def evaluate(*args):
        evaluations.append(1)
        return original_evaluate(*args)
    monkeypatch.setattr(pipeline, 'evaluate_quality', evaluate)
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
    assert len(evaluations) == 4


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


def test_filter_export_includes_both_midi_sets_and_report(pipeline_env, tmp_path):
    from music_re.core.export import export_bundle
    pipeline, ws, source, _ = pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='filter')
    manifest = ws.get_manifest(source)
    target = export_bundle(ws, source, tmp_path / 'bundle')
    for name in manifest.midis:
        assert (target / 'midis' / f'{name}.mid').read_bytes() == Path(manifest.validated_midis[name]).read_bytes()
        assert (target / 'midis_original' / f'{name}.mid').read_bytes() == Path(manifest.midis[name]).read_bytes()
    assert (target / 'spectral_validation.json').is_file()


def test_config_change_only_reruns_spectral_and_evaluation(pipeline_env, monkeypatch):
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


def test_legacy_manifest_reuses_upstream_fingerprints(pipeline_env):
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
