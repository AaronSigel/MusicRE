import pytest
import soundfile as sf
import numpy as np
from pathlib import Path

from music_re.core.analyzer import analyze_audio
from music_re.core.audio_io import decode_audio, normalize_audio

def test_stereo_loudness_not_cancelled_by_antiphase(tmp_path):
    # Stereo anti-phase: left = x, right = -x
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    stereo = np.stack([x, -x], axis=-1)
    
    wav_path = tmp_path / "antiphase.wav"
    sf.write(str(wav_path), stereo, sr)
    
    analysis = analyze_audio(wav_path)
    # If downmixed to mono, x + (-x) = 0 -> -70 LUFS.
    # On true stereo it should be around -9 to -12 LUFS.
    assert analysis.lufs > -30.0

def test_silent_audio_returns_na_key(tmp_path):
    sr = 44100
    silence = np.zeros((sr, 2), dtype=np.float32)
    wav_path = tmp_path / "silence.wav"
    sf.write(str(wav_path), silence, sr)
    
    analysis = analyze_audio(wav_path)
    assert analysis.key == "N/A"
    assert analysis.key_confidence == 0.0

def test_reject_zero_length_audio(tmp_path):
    sr = 44100
    empty = np.zeros((0, 2), dtype=np.float32)
    wav_path = tmp_path / "empty.wav"
    sf.write(str(wav_path), empty, sr)
    
    with pytest.raises(ValueError, match="empty|zero-length"):
        decode_audio(wav_path)

def test_reject_nan_inf_audio(tmp_path):
    sr = 44100
    bad = np.array([[np.nan, 0.0], [0.1, np.inf]], dtype=np.float32)
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_audio(bad)
