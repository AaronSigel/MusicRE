from pathlib import Path
import pytest

from music_re.schemas import WorkspaceManifest
from music_re.workspace import WorkspaceManager

def test_workspace_manifest_initialization(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track = tmp_path / "song.wav"
    track.write_bytes(b"TEST_AUDIO_FOR_MANIFEST")

    manifest = ws.get_manifest(track)
    assert isinstance(manifest, WorkspaceManifest)
    assert manifest.status == "pending"
    assert manifest.completed_stages == []

def test_workspace_manifest_stage_updates(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track = tmp_path / "song.wav"
    track.write_bytes(b"TEST_AUDIO_FOR_UPDATES")

    ws.update_manifest(track, status="processing", current_stage="separation")
    m = ws.get_manifest(track)
    assert m.status == "processing"
    assert m.current_stage == "separation"

    ws.mark_stage_complete(track, "separation", artifacts={"vocals": "/path/to/vocals.wav"})
    m2 = ws.get_manifest(track)
    assert "separation" in m2.completed_stages
    assert m2.stems.get("vocals") == "/path/to/vocals.wav"

def test_workspace_manifest_error_recording(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track = tmp_path / "song.wav"
    track.write_bytes(b"TEST_AUDIO_FOR_ERROR")

    ws.record_stage_error(track, stage="transcription", error_message="Basic pitch ONNX parse failed")
    m = ws.get_manifest(track)
    assert m.status == "failed"
    assert "transcription" in m.stage_errors
    assert "ONNX parse failed" in m.stage_errors["transcription"]
