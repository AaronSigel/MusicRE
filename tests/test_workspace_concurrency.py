import json
import multiprocessing
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_re.workspace import WorkspaceManager

N_OPS = 50


def _make_source(tmp_path):
    source = tmp_path / 'song.wav'
    sf.write(source, np.zeros(2205), 22050)
    return source


def _stage_worker(cache_root, source, tag, errors):
    try:
        ws = WorkspaceManager(cache_root)
        for i in range(N_OPS):
            ws.mark_stage_complete(source, f'{tag}_{i}')
            ws.update_manifest(source, status='processing', current_stage=f'{tag}_{i}')
    except BaseException as e:  # Surface child failures to the parent.
        errors.put(f'{tag}: {type(e).__name__}: {e}')


def _config_worker(cache_root, source, tag, errors):
    try:
        ws = WorkspaceManager(cache_root)
        for i in range(N_OPS):
            with ws.lock(source):
                config = dict(ws.get_manifest(source).run_config)
                config[f'{tag}_{i}'] = i
                ws.update_manifest(source, run_config=config)
    except BaseException as e:
        errors.put(f'{tag}: {type(e).__name__}: {e}')


def _run_two(target, cache_root, source):
    ctx = multiprocessing.get_context('fork')
    errors = ctx.Queue()
    procs = [ctx.Process(target=target, args=(cache_root, source, tag, errors)) for tag in ('a', 'b')]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    for p in procs:
        if p.is_alive():
            p.kill()
            pytest.fail('worker did not finish (deadlock?)')
    found = []
    while not errors.empty():
        found.append(errors.get())
    return found


def _assert_no_temp_files(track_dir: Path):
    leftovers = sorted(p.name for p in track_dir.iterdir()
                       if p.is_file() and p.name not in {'manifest.json', '.lock'})
    assert leftovers == []


def test_two_processes_mark_stage_complete_lose_nothing(tmp_path):
    source = _make_source(tmp_path)
    cache_root = tmp_path / 'cache'
    ws = WorkspaceManager(cache_root)
    ws.get_manifest(source)
    errors = _run_two(_stage_worker, cache_root, source)
    assert errors == []
    m_path = ws.get_manifest_path(source)
    data = json.loads(m_path.read_text(encoding='utf-8'))
    expected = {f'{tag}_{i}' for tag in ('a', 'b') for i in range(N_OPS)}
    assert expected <= set(data['completed_stages'])
    _assert_no_temp_files(m_path.parent)


def test_two_processes_locked_update_manifest_lose_nothing(tmp_path):
    source = _make_source(tmp_path)
    cache_root = tmp_path / 'cache'
    ws = WorkspaceManager(cache_root)
    ws.get_manifest(source)
    errors = _run_two(_config_worker, cache_root, source)
    assert errors == []
    data = json.loads(ws.get_manifest_path(source).read_text(encoding='utf-8'))
    assert data['run_config'] == {f'{tag}_{i}': i for tag in ('a', 'b') for i in range(N_OPS)}
    _assert_no_temp_files(ws.get_manifest_path(source).parent)


def test_nested_lock_does_not_deadlock(tmp_path):
    source = _make_source(tmp_path)
    ws = WorkspaceManager(tmp_path / 'cache')
    done = threading.Event()

    def body():
        with ws.lock(source):
            with ws.lock(source):
                ws.mark_stage_complete(source, 'separation', {'vocals': 'x.wav'})
                ws.invalidate_stages(source, ['analysis'])
            ws.record_stage_error(source, 'analysis', 'boom')
        done.set()

    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(timeout=20)
    assert done.is_set(), 'nested lock deadlocked'
    m = ws.get_manifest(source)
    assert m.stems == {'vocals': 'x.wav'}
    assert m.stage_errors == {'analysis': 'boom'}


def test_lock_is_exclusive_across_processes(tmp_path):
    source = _make_source(tmp_path)
    ws = WorkspaceManager(tmp_path / 'cache')
    ws.get_manifest(source)
    ctx = multiprocessing.get_context('fork')
    acquired = ctx.Event()
    with ws.lock(source):
        p = ctx.Process(target=_acquire_and_signal, args=(tmp_path / 'cache', source, acquired))
        p.start()
        assert not acquired.wait(1.0), 'second process entered a held lock'
    assert acquired.wait(20), 'second process never acquired released lock'
    p.join(timeout=20)


def _acquire_and_signal(cache_root, source, acquired):
    with WorkspaceManager(cache_root).lock(source):
        acquired.set()


@pytest.fixture
def completed_pipeline(tmp_path, monkeypatch):
    """Run the real pipeline (balanced, evaluation on) with fake model stages."""
    from types import SimpleNamespace

    import pretty_midi

    import music_re.core.pipeline as pipeline
    from music_re.core.backends.transcriber_base import save_transcription
    from music_re.schemas import MusicalTimeline, StemSeparationResult, TrackAnalysis

    ws = WorkspaceManager(tmp_path / 'cache')
    source = tmp_path / 'song.wav'
    sf.write(source, np.zeros(22050), 22050)

    def separate(file_path, workspace, six_stems, **kwargs):
        stems = {}
        for name in ['vocals', 'drums', 'bass', 'other']:
            p = workspace.get_stems_dir(file_path, 'fake') / f'{name}.wav'
            sf.write(p, np.zeros((22050, 2)), 22050)
            stems[name] = str(p)
        return StemSeparationResult(workspace_dir=str(ws.get_track_dir(source)), model_used='fake', stems=stems)

    def transcribe(audio_path, output_midi_path, **kwargs):
        pm = pretty_midi.PrettyMIDI()
        return save_transcription(pm, output_midi_path, pm if kwargs['save_raw'] else None, 'fake', {})

    monkeypatch.setattr(pipeline, 'separate_stems', separate)
    monkeypatch.setattr(pipeline, 'analyze_audio', lambda *a, **k: TrackAnalysis(
        file_path=str(source), duration_sec=1, bpm=120, key_confidence=0, lufs=-70, beat_times=[]))
    monkeypatch.setattr(pipeline, 'get_structure_analyzer', lambda **kwargs: SimpleNamespace(
        analyze=lambda *a, **k: MusicalTimeline(duration_sec=1, bpm=120, beats=[0, .5], downbeats=[])))
    monkeypatch.setattr(pipeline, 'get_transcriber_backend', lambda **kwargs: SimpleNamespace(transcribe=transcribe))
    pipeline.run_pipeline(source, workspace=ws, profile='balanced')
    return pipeline, ws, source


def test_standalone_reevaluation_keeps_bundle_valid(completed_pipeline):
    from music_re.core.export import validate_bundle
    from music_re.core.quality import evaluate_quality

    pipeline, ws, source = completed_pipeline
    validate_bundle(ws, source, 'balanced')
    report = evaluate_quality(ws, source).model_copy(update={'evaluated_at': '2099-01-01T00:00:00'})
    path = ws.save_quality_report(source, report)
    validate_bundle(ws, source, 'balanced')
    manifest = ws.get_manifest(source)
    assert manifest.artifact_hashes['evaluation'] == pipeline._hashes([path])
    assert 'evaluation' in manifest.completed_stages
