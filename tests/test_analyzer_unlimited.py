from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from music_re.core.analyzer import analyze_audio

def test_analyzer_unlimited_beats(tmp_path: Path):
    sr = 22050
    # 40 seconds at 120 bpm = 80 beats (>64)
    duration_sec = 40.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # 2 Hz pulses = 120 bpm
    y = 0.5 * np.sin(2 * np.pi * 2.0 * t) * (np.sin(2 * np.pi * 2.0 * t) > 0.95)
    test_wav = tmp_path / "long_track.wav"
    sf.write(str(test_wav), y, sr)

    analysis = analyze_audio(test_wav)
    assert len(analysis.beat_times) > 64, f"Expected >64 beats, got {len(analysis.beat_times)}"

def test_analyzer_skip_drums_key(tmp_path: Path):
    sr = 22050
    y = np.random.uniform(-0.1, 0.1, sr * 2).astype(np.float32)
    drums_wav = tmp_path / "drums.wav"
    sf.write(str(drums_wav), y, sr)

    # 1. By filename
    res_by_name = analyze_audio(drums_wav)
    assert res_by_name.key in ("N/A", None)
    assert res_by_name.key_confidence == 0.0

    # 2. By explicit stem_name
    other_wav = tmp_path / "track_stem.wav"
    sf.write(str(other_wav), y, sr)
    res_by_param = analyze_audio(other_wav, stem_name="drums")
    assert res_by_param.key in ("N/A", None)
    assert res_by_param.key_confidence == 0.0
