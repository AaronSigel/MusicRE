from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import pytest
from typer.testing import CliRunner

from music_re.cli import app
from music_re.schemas import MusicalTimeline, TimelineSection
from music_re.workspace import WorkspaceManager

runner = CliRunner()

def test_cli_to_midi_with_timeline(tmp_path: Path):
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    stem_wav = tmp_path / "vocals.wav"
    sf.write(str(stem_wav), y, sr)

    tl = MusicalTimeline(
        duration_sec=1.0,
        bpm=120.0,
        beats=[0.0, 0.5, 1.0],
        downbeats=[0.0],
        sections=[TimelineSection(label="intro", start_sec=0.0, end_sec=1.0, confidence=1.0)],
    )
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    tl_file = ws.save_timeline(stem_wav, tl)

    out_midi = tmp_path / "vocals_test.mid"
    res = runner.invoke(app, ["to-midi", str(stem_wav), "--out", str(out_midi), "--timeline", str(tl_file)])
    assert res.exit_code == 0
    assert out_midi.exists()

def test_deconstruct_generates_all_stems_project(tmp_path: Path):
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    y = 0.4 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    track_wav = tmp_path / "track.wav"
    sf.write(str(track_wav), y, sr)

    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_dir = ws.get_track_dir(track_wav)

    # Mock separation to avoid heavy Demucs in unit test
    stems_dir = ws.get_stems_dir(track_wav, model_name="htdemucs")
    for s in ["vocals", "drums", "bass", "other"]:
        sf.write(str(stems_dir / f"{s}.wav"), y, sr)
    (stems_dir / ".complete_htdemucs").write_text("ok")

    with patch("music_re.cli.WorkspaceManager", return_value=ws):
        res = runner.invoke(app, ["deconstruct", str(track_wav)])
        assert res.exit_code == 0

    all_stems_mid = track_dir / "midis" / "all_stems.mid"
    assert all_stems_mid.exists()

    manifest = ws.get_manifest(track_wav)
    assert "all_stems" in manifest.midis
    assert manifest.midis["all_stems"] == str(all_stems_mid)
