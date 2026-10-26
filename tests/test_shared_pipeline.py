import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from typer.testing import CliRunner

from music_re.schemas import MusicalTimeline, TrackAnalysis, StemSeparationResult
from music_re.workspace import WorkspaceManager


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    import music_re.core.pipeline as pipeline
    from music_re.core.backends.transcriber_base import save_transcription
    ws = WorkspaceManager(tmp_path / 'cache')
    source = tmp_path / 'song.wav'
    sf.write(source, np.zeros(22050), 22050)
    counts = dict(separation=0, transcription=0, structure=0)
    def separate(file_path, workspace, six_stems, **kwargs):
        counts['separation'] += 1
        names = ['vocals', 'drums', 'bass', 'other'] + (['guitar', 'piano'] if six_stems else [])
        stems = {}
        for name in names:
            p = workspace.get_stems_dir(file_path, 'fake') / f'{name}.wav'
            sf.write(p, np.zeros((44100, 2)), 44100)
            stems[name] = str(p)
        return StemSeparationResult(workspace_dir=str(ws.get_track_dir(source)), model_used='fake', stems=stems)
    def structure(*args, **kwargs):
        counts['structure'] += 1
        return MusicalTimeline(duration_sec=1, bpm=90, beats=[0, 2/3], downbeats=[])
    def transcribe(audio_path, output_midi_path, **kwargs):
        counts['transcription'] += 1
        pm = pretty_midi.PrettyMIDI()
        return save_transcription(pm, output_midi_path, pm if kwargs['save_raw'] else None, 'fake', {})
    monkeypatch.setattr(pipeline, 'separate_stems', separate)
    monkeypatch.setattr(pipeline, 'analyze_audio', lambda *a, **k: TrackAnalysis(file_path=str(source), duration_sec=1, bpm=120, key_confidence=0, lufs=-70, beat_times=[]))
    monkeypatch.setattr(pipeline, 'get_structure_analyzer', lambda **kwargs: SimpleNamespace(analyze=structure))
    monkeypatch.setattr(pipeline, 'get_transcriber_backend', lambda **kwargs: SimpleNamespace(transcribe=transcribe))
    return pipeline, ws, source, counts


def test_pipeline_resumes_and_regenerates_missing_midi(pipeline_env):
    p, ws, source, counts = pipeline_env
    first = p.run_pipeline(source, workspace=ws, profile='fast')
    assert ws.get_manifest(source).status == 'completed'
    assert set(first['midis']) == {'vocals', 'drums', 'bass', 'other', 'all_stems'}
    p.run_pipeline(source, workspace=ws, profile='fast')
    assert counts == dict(separation=1, structure=1, transcription=4)
    Path(first['midis']['vocals']).unlink()
    p.run_pipeline(source, workspace=ws, profile='fast')
    assert counts['transcription'] == 5
    assert ws.get_manifest(source).analysis['bpm'] == 120
    assert ws.get_manifest(source).run_config['profile'] == 'fast'


def test_pipeline_switches_six_to_four_without_stale_artifacts(pipeline_env):
    p, ws, source, counts = pipeline_env
    p.run_pipeline(source, workspace=ws, profile='balanced', six_stems=True)
    p.run_pipeline(source, workspace=ws, profile='fast', six_stems=True)
    m = ws.get_manifest(source)
    assert set(m.stems) == {'vocals', 'drums', 'bass', 'other'}
    assert 'guitar' not in m.midis
    assert m.evaluation is None
    assert 'evaluation' not in m.completed_stages


def test_pipeline_records_evaluation_failure_at_correct_stage(pipeline_env, monkeypatch):
    p, ws, source, _ = pipeline_env
    def fail(*args, **kwargs):
        raise RuntimeError('quality failed')
    monkeypatch.setattr(p, 'evaluate_quality', fail)
    with pytest.raises(RuntimeError, match='quality failed'):
        p.run_pipeline(source, workspace=ws)
    m = ws.get_manifest(source)
    assert m.status == 'failed'
    assert m.current_stage == 'evaluation'
    assert m.stage_errors == {'evaluation': 'quality failed'}


def test_invalid_profile_does_not_create_track_workspace(tmp_path):
    from music_re.core.pipeline import run_pipeline
    ws = WorkspaceManager(tmp_path / 'cache')
    with pytest.raises(ValueError, match='profile'):
        run_pipeline(tmp_path / 'missing.mp3', workspace=ws, profile='invalid')
    assert list(ws.cache_root.iterdir()) == []


def test_cli_and_mcp_share_pipeline(pipeline_env, monkeypatch):
    from music_re.cli import app
    import music_re.cli as cli
    import music_re.mcp_server as server
    _, ws, source, counts = pipeline_env
    monkeypatch.setattr(cli, 'WorkspaceManager', lambda **kwargs: ws)
    monkeypatch.setattr(server, 'WorkspaceManager', lambda **kwargs: ws)
    result = CliRunner().invoke(app, ['deconstruct', str(source), '--profile', 'fast'])
    assert result.exit_code == 0, result.output
    response = asyncio.run(server.deconstruct_full_pipeline(str(source), profile='fast'))
    assert response['midis'] == ws.get_manifest(source).midis
    assert counts['separation'] == 1


def test_profile_change_invalidates_transcription_settings(pipeline_env):
    p, ws, source, counts = pipeline_env
    p.run_pipeline(source, workspace=ws, profile='balanced')
    p.run_pipeline(source, workspace=ws, profile='quality')
    assert counts['separation'] == 2
    assert counts['structure'] == 2
    assert counts['transcription'] == 8
    assert ws.get_manifest(source).run_config['structure_model'] == 'harmonix-all'


def test_changed_stem_bytes_invalidate_downstream(pipeline_env):
    p, ws, source, counts = pipeline_env
    first = p.run_pipeline(source, workspace=ws, profile='fast')
    sf.write(first['stems']['vocals'], np.ones((44100, 2)) * .1, 44100)
    p.run_pipeline(source, workspace=ws, profile='fast')
    assert counts['structure'] == 2
    assert counts['transcription'] == 8


def test_partial_transcription_resumes_successes(pipeline_env, monkeypatch):
    p, ws, source, counts = pipeline_env
    original = p.get_transcriber_backend
    def backend(**kwargs):
        if kwargs['mode'] == 'drums':
            def fail(*args, **kwargs):
                raise RuntimeError('drum failed')
            return SimpleNamespace(transcribe=fail)
        return original(**kwargs)
    monkeypatch.setattr(p, 'get_transcriber_backend', backend)
    with pytest.raises(RuntimeError, match='drum failed'):
        p.run_pipeline(source, workspace=ws, profile='fast')
    assert set(ws.get_manifest(source).transcription_results) == {'vocals'}
    monkeypatch.setattr(p, 'get_transcriber_backend', original)
    p.run_pipeline(source, workspace=ws, profile='fast')
    assert counts['transcription'] == 4


def test_checkpoint_change_invalidates_artifacts_before_attribution(pipeline_env, monkeypatch):
    p, ws, source, counts = pipeline_env
    checkpoint = ['old-checkpoint']
    monkeypatch.setattr(p, 'model_inventory', lambda *args: {'separator': [{'sha256': checkpoint[0]}], 'tonal': [], 'structure': []})
    p.run_pipeline(source, workspace=ws, profile='fast')
    old_fingerprint = ws.get_manifest(source).stage_fingerprints['separation']
    checkpoint[0] = 'new-checkpoint'
    p.run_pipeline(source, workspace=ws, profile='fast')
    manifest = ws.get_manifest(source)
    assert counts['separation'] == 2
    assert counts['transcription'] == 8
    assert manifest.stage_fingerprints['separation'] != old_fingerprint
    assert manifest.runtime['weights']['separator'][0]['sha256'] == 'new-checkpoint'
