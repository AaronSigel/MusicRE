from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from music_re.core.audio_io import decode_audio, decode_to_temp_wav, normalize_audio

@pytest.fixture
def sample_wav(tmp_path):
    p = tmp_path / "sample.wav"
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    stereo = np.stack([data, data], axis=-1).astype(np.float32)
    sf.write(str(p), stereo, sr, subtype="PCM_16")
    return p, sr, stereo

def test_decode_to_temp_wav(sample_wav):
    src_p, sr, orig = sample_wav
    with decode_to_temp_wav(src_p) as temp_wav:
        assert Path(temp_wav).exists()
        data, read_sr = sf.read(str(temp_wav), dtype="float32")
        assert read_sr == sr
        assert data.shape[0] == orig.shape[0]
    assert not Path(temp_wav).exists()

def test_decode_audio(sample_wav):
    src_p, sr, orig = sample_wav
    data, out_sr = decode_audio(src_p, target_sr=22050, normalize_db=-1.0)
    assert out_sr == 22050
    assert data.dtype == np.float32
    # -1 dB is ~0.891 peak
    peak = np.max(np.abs(data))
    assert 0.88 <= peak <= 0.90

def test_normalize_audio():
    data = np.array([0.1, -0.5, 0.25], dtype=np.float32)
    norm = normalize_audio(data, target_peak_db=-1.0)
    peak = np.max(np.abs(norm))
    assert pytest.approx(peak, rel=1e-3) == 10 ** (-1.0 / 20.0)

def test_decode_nonexistent_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        decode_audio(tmp_path / "missing.wav")
