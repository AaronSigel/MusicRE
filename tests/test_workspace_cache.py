from pathlib import Path
from music_re.workspace import compute_file_content_hash, WorkspaceManager

def test_compute_file_content_hash(tmp_path: Path):
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.wav"
    # Files identical first 64KB, differ after 65KB
    prefix = b"X" * (65 * 1024)
    f1.write_bytes(prefix + b"DATA_ONE")
    f2.write_bytes(prefix + b"DATA_TWO")

    h1 = compute_file_content_hash(f1)
    h2 = compute_file_content_hash(f2)
    assert len(h1) == 16
    assert len(h2) == 16
    assert h1 != h2, "Content hashing must read full file to avoid prefix collisions"

def test_segregated_model_cache(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    dummy = tmp_path / "track.wav"
    dummy.write_bytes(b"TEST_AUDIO_DATA_FOR_SEGREGATION")

    stems_4 = ws.get_stems_dir(dummy, model_name="htdemucs")
    stems_6 = ws.get_stems_dir(dummy, model_name="htdemucs_6s")

    assert stems_4 != stems_6
    assert "htdemucs" in str(stems_4)
    assert "htdemucs_6s" in str(stems_6)

    stem_v4 = ws.get_stem_path(dummy, "vocals", model_name="htdemucs")
    stem_v6 = ws.get_stem_path(dummy, "vocals", model_name="htdemucs_6s")
    assert stem_v4 != stem_v6


def test_switching_stem_sets_invalidates_dependents(tmp_path):
    ws = WorkspaceManager(tmp_path / 'cache')
    audio = tmp_path / 'song.wav'
    audio.write_bytes(b'input')
    six = {n: f'/six/{n}.wav' for n in ('vocals', 'bass', 'drums', 'other', 'guitar', 'piano')}
    four = {n: f'/four/{n}.wav' for n in ('vocals', 'bass', 'drums', 'other')}
    ws.mark_stage_complete(audio, 'separation', six)
    ws.mark_stage_complete(audio, 'analysis', {'bpm': 120})
    ws.mark_stage_complete(audio, 'structure', {'timeline': {'bpm': 120}})
    ws.mark_stage_complete(audio, 'transcription', {'guitar': '/old.mid'})
    ws.mark_stage_complete(audio, 'evaluation', {'report': {}})
    ws.mark_stage_complete(audio, 'separation', four)
    m = ws.get_manifest(audio)
    assert m.stems == four
    assert m.midis == {}
    assert m.timeline is None and m.analysis is None and m.evaluation is None
    assert m.completed_stages == ['separation']
    ws.mark_stage_complete(audio, 'separation', six)
    assert ws.get_manifest(audio).stems == six


def test_invalidated_reports_cannot_be_loaded_from_old_files(tmp_path):
    from music_re.schemas import MusicalTimeline
    ws = WorkspaceManager(tmp_path / 'cache')
    audio = tmp_path / 'song.wav'
    audio.write_bytes(b'input')
    ws.save_timeline(audio, MusicalTimeline(duration_sec=1, bpm=120, beats=[], downbeats=[]))
    ws.mark_stage_complete(audio, 'separation', {'vocals': '/new.wav'})
    assert ws.load_timeline(audio) is None
