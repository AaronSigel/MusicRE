import json
from pathlib import Path
from music_re.workspace import compute_file_hash, WorkspaceManager

def test_workspace_hash_and_cache(tmp_path: Path):
    dummy_file = tmp_path / "song.wav"
    dummy_file.write_bytes(b"RIFF" + b"\x00" * 1000)
    file_hash = compute_file_hash(dummy_file)
    assert len(file_hash) == 8

    cache_dir = tmp_path / ".cache"
    ws = WorkspaceManager(cache_root=cache_dir)
    track_dir = ws.get_track_dir(dummy_file)
    assert track_dir.exists()
    assert file_hash in track_dir.name
    assert "song" in track_dir.name

    # Check metadata save and load
    meta = {"bpm": 124.0, "key": "A minor"}
    ws.save_metadata(dummy_file, meta)
    loaded = ws.load_metadata(dummy_file)
    assert loaded == meta

    # Check stem paths
    stems_dir = ws.get_stems_dir(dummy_file)
    assert stems_dir.exists()
    stem_file = ws.get_stem_path(dummy_file, "vocals")
    assert stem_file == stems_dir / "vocals.wav"
