from pathlib import Path
import numpy as np
import soundfile as sf
import pytest
from typer.testing import CliRunner

from music_re.cli import app
from music_re.schemas import MusicalTimeline, TimelineSection
from music_re.workspace import WorkspaceManager

runner = CliRunner()

def test_workspace_save_and_load_timeline(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track = tmp_path / "test.wav"
    track.write_bytes(b"TEST_FOR_TIMELINE")

    tl = MusicalTimeline(
        duration_sec=16.0,
        bpm=124.0,
        beats=[0.0, 0.48, 0.96, 1.45],
        downbeats=[0.0],
        sections=[TimelineSection(label="intro", start_sec=0.0, end_sec=16.0, confidence=0.9)],
    )

    timeline_file = ws.save_timeline(track, tl)
    assert timeline_file.exists()
    assert timeline_file.name == "timeline.json"

    loaded = ws.load_timeline(track)
    assert loaded is not None
    assert loaded.bpm == 124.0
    assert len(loaded.sections) == 1
    assert loaded.sections[0].label == "intro"

    manifest = ws.get_manifest(track)
    assert manifest.timeline is not None
    assert manifest.timeline["bpm"] == 124.0

def test_cli_structure_command(tmp_path: Path):
    sr = 22050
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    test_wav = tmp_path / "sample.wav"
    sf.write(str(test_wav), y, sr)

    res = runner.invoke(app, ["structure", str(test_wav), "--backend", "librosa"])
    assert res.exit_code == 0
    assert "Structure Analysis" in res.stdout
    assert "BPM" in res.stdout

    res_json = runner.invoke(app, ["structure", str(test_wav), "--json", "--backend", "librosa"])
    assert res_json.exit_code == 0
    assert '"bpm"' in res_json.stdout
    assert '"sections"' in res_json.stdout
