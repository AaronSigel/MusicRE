from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from music_re.core.backends import get_structure_analyzer
from music_re.core.separator import separate_stems
from music_re.schemas import MusicalTimeline
from music_re.workspace import WorkspaceManager


@pytest.fixture
def test_audio_with_stems(tmp_path: Path):
    sr = 22050
    duration_sec = 6.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # 2 Hz beat pulse
    pulse = 0.5 * np.sin(2 * np.pi * 2.0 * t) * (np.sin(2 * np.pi * 2.0 * t) > 0.9)
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    mix = (pulse + tone).astype(np.float32)

    track_wav = tmp_path / "song.wav"
    sf.write(str(track_wav), mix, sr)

    # Pre-generate mock stems in workspace to test stem reuse without re-running Demucs
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    stems_dir = ws.get_stems_dir(track_wav, model_name="htdemucs")
    stems = {}
    for s in ["vocals", "drums", "bass", "other"]:
        p = stems_dir / f"{s}.wav"
        sf.write(str(p), pulse if s == "drums" else tone, sr)
        stems[s] = str(p)
    (stems_dir / ".complete_htdemucs").write_text("ok")

    return track_wav, ws, stems


def test_structure_e2e_timeline_generation(test_audio_with_stems):
    track_wav, ws, stems = test_audio_with_stems

    analyzer = get_structure_analyzer(name="librosa")
    timeline = analyzer.analyze(track_wav, stems=stems)

    assert isinstance(timeline, MusicalTimeline)
    assert timeline.duration_sec == pytest.approx(6.0, abs=0.1)
    assert timeline.bpm > 0
    assert len(timeline.beats) > 0
    assert len(timeline.downbeats) > 0
    assert len(timeline.sections) > 0

    # Save to workspace
    tl_path = ws.save_timeline(track_wav, timeline)
    assert tl_path.exists()
    assert tl_path.name == "timeline.json"

    # Verify manifest updated
    m = ws.get_manifest(track_wav)
    assert "structure" in m.completed_stages
    assert m.timeline is not None
    assert m.timeline["bpm"] == timeline.bpm
    assert len(m.timeline["sections"]) == len(timeline.sections)


def test_structure_e2e_fallback_on_missing_stems(tmp_path: Path):
    sr = 22050
    duration_sec = 4.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    mix = 0.5 * np.sin(2 * np.pi * 330.0 * t).astype(np.float32)

    track_wav = tmp_path / "no_stems.wav"
    sf.write(str(track_wav), mix, sr)

    analyzer = get_structure_analyzer(name="auto")
    # Pass None for stems -> must analyze full track gracefully
    timeline = analyzer.analyze(track_wav, stems=None)
    assert isinstance(timeline, MusicalTimeline)
    assert timeline.duration_sec == pytest.approx(4.0, abs=0.1)
    assert len(timeline.sections) > 0
