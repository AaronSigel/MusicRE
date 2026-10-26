from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from music_re.core.slicer import slice_audio

def test_slice_audio_valid(tmp_path: Path):
    sr = 22050
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    in_file = tmp_path / "sine.wav"
    sf.write(str(in_file), audio, sr)

    out_file = tmp_path / "slice.wav"
    res = slice_audio(in_file, out_file, start_sec=1.0, end_sec=2.5)
    assert res == out_file
    assert out_file.exists()

    data, out_sr = sf.read(str(out_file))
    assert out_sr == sr
    expected_len = int(1.5 * sr)
    assert abs(len(data) - expected_len) <= 5

def test_slice_audio_end_beyond_duration(tmp_path: Path):
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    in_file = tmp_path / "short.wav"
    sf.write(str(in_file), audio, sr)

    out_file = tmp_path / "slice_end.wav"
    slice_audio(in_file, out_file, start_sec=1.0, end_sec=5.0)
    data, out_sr = sf.read(str(out_file))
    expected_len = int(1.0 * sr)
    assert abs(len(data) - expected_len) <= 5

def test_slice_audio_invalid_range(tmp_path: Path):
    in_file = tmp_path / "empty.wav"
    sf.write(str(in_file), np.zeros(22050), 22050)
    out_file = tmp_path / "err.wav"

    with pytest.raises(ValueError):
        slice_audio(in_file, out_file, start_sec=2.0, end_sec=1.0)
