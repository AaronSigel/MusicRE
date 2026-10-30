import numpy as np
import pytest

from music_re.core.chords import detect_chords_for_bars
from music_re.schemas import BarChord, MusicalTimeline


def _generate_sine(freq: float, duration: float, sr: int = 22050) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_detect_chords_c_major():
    sr = 22050
    dur = 2.0
    # C4: 261.63, E4: 329.63, G4: 392.00
    y_c = _generate_sine(261.63, dur, sr) + _generate_sine(329.63, dur, sr) + _generate_sine(392.00, dur, sr)

    bars = [{"start_sec": 0.0, "end_sec": 2.0}]
    results = detect_chords_for_bars(y_c, sr, bars)

    assert len(results) == 1
    assert results[0]["bar"] == 1
    assert results[0]["chord"] == "C"
    assert results[0]["confidence"] > 0.8
    assert results[0]["start_sec"] == 0.0
    assert results[0]["end_sec"] == 2.0


def test_detect_chords_a_minor():
    sr = 22050
    dur = 2.0
    # A3: 220.00, C4: 261.63, E4: 329.63
    y_am = _generate_sine(220.00, dur, sr) + _generate_sine(261.63, dur, sr) + _generate_sine(329.63, dur, sr)

    bars = [(0.0, 2.0)]
    results = detect_chords_for_bars(y_am, sr, bars)

    assert len(results) == 1
    assert results[0]["bar"] == 1
    assert results[0]["chord"] == "Am"
    assert results[0]["confidence"] > 0.8


def test_detect_chords_silence():
    sr = 22050
    dur = 2.0
    y_silent = np.zeros(int(sr * dur), dtype=np.float32)

    bars = [{"start_sec": 0.0, "end_sec": 2.0}]
    results = detect_chords_for_bars(y_silent, sr, bars)

    assert len(results) == 1
    assert results[0]["bar"] == 1
    assert results[0]["chord"] == "N/A"
    assert results[0]["confidence"] == 0.0


def test_detect_chords_progression_multi_bars():
    sr = 22050
    bar_dur = 1.0
    # Bar 1: C major (C4, E4, G4)
    b1 = _generate_sine(261.63, bar_dur, sr) + _generate_sine(329.63, bar_dur, sr) + _generate_sine(392.00, bar_dur, sr)
    # Bar 2: Am (A3, C4, E4)
    b2 = _generate_sine(220.00, bar_dur, sr) + _generate_sine(261.63, bar_dur, sr) + _generate_sine(329.63, bar_dur, sr)
    # Bar 3: Silence
    b3 = np.zeros(int(sr * bar_dur), dtype=np.float32)

    y = np.concatenate([b1, b2, b3])
    bars = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

    results = detect_chords_for_bars(y, sr, bars)
    assert len(results) == 3

    assert results[0]["bar"] == 1
    assert results[0]["chord"] == "C"

    assert results[1]["bar"] == 2
    assert results[1]["chord"] == "Am"

    assert results[2]["bar"] == 3
    assert results[2]["chord"] == "N/A"
    assert results[2]["confidence"] == 0.0


def test_musical_timeline_barchord_schema():
    b = BarChord(bar=1, start_sec=0.0, end_sec=2.0, chord="C", confidence=0.99)
    tl = MusicalTimeline(
        duration_sec=2.0,
        bpm=120.0,
        beats=[0.0, 0.5, 1.0, 1.5],
        downbeats=[0.0],
        chords=[b],
    )
    dumped = tl.model_dump()
    assert len(dumped["chords"]) == 1
    assert dumped["chords"][0]["chord"] == "C"


def test_dawproject_chord_markers(tmp_path):
    from dawproject import DawProject
    import soundfile as sf
    from music_re.core.dawproject_export import export_dawproject
    from music_re.workspace import WorkspaceManager

    ws = WorkspaceManager(tmp_path / "cache")
    source = tmp_path / "test.wav"
    sr = 22050
    sf.write(str(source), np.zeros(sr * 4, dtype=np.float32), sr)

    stems = {}
    stems_dir = ws.get_stems_dir(source, "fixture")
    for role in ("vocals", "bass", "drums", "other"):
        sp = stems_dir / f"{role}.wav"
        sf.write(str(sp), np.zeros(sr * 4, dtype=np.float32), sr)
        stems[role] = str(sp)
    ws.mark_stage_complete(source, "separation", stems)

    timeline = MusicalTimeline(
        duration_sec=4.0,
        bpm=120.0,
        beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        downbeats=[0.0, 2.0],
        chords=[
            BarChord(bar=1, start_sec=0.0, end_sec=2.0, chord="C", confidence=0.95),
            BarChord(bar=2, start_sec=2.0, end_sec=4.0, chord="Am", confidence=0.92),
        ],
    )
    ws.save_timeline(source, timeline)

    out_file = tmp_path / "exported.dawproject"
    res = export_dawproject(ws, source, out_file)
    assert res.is_file()

    project = DawProject.load_project(str(out_file))
    assert project.arrangement.markers is not None
    marker_names = [m.name for m in project.arrangement.markers.markers]
    assert marker_names == ["C", "Am"]
