from pathlib import Path
import numpy as np
import pytest

from music_re.core.audio_io import normalize_audio
from music_re.schemas import WorkspaceManifest
from music_re.workspace import WorkspaceManager

def test_normalize_audio_empty_array():
    empty = np.array([], dtype=np.float32)
    res = normalize_audio(empty)
    assert res.size == 0

def test_manifest_resumption_differentiates_four_and_six_stems(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track = tmp_path / "song.wav"
    track.write_bytes(b"TEST_FOR_6_STEMS_RESUME")

    # Simulate 4-stem completion
    four_stems = {
        "vocals": str(tmp_path / "v.wav"),
        "drums": str(tmp_path / "d.wav"),
        "bass": str(tmp_path / "b.wav"),
        "other": str(tmp_path / "o.wav"),
    }
    for p in four_stems.values():
        Path(p).write_bytes(b"RIFF\x00\x00")

    ws.mark_stage_complete(track, "separation", artifacts=four_stems)
    m = ws.get_manifest(track)
    assert len(m.stems) == 4

    # Verification helper logic for CLI: if requesting 6 stems, 4-stem manifest cannot satisfy it
    expected_count = 6
    can_resume_6 = len(m.stems) == expected_count and all(Path(p).is_file() for p in m.stems.values())
    assert not can_resume_6
