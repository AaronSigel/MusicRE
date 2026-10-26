from pathlib import Path
import numpy as np
import soundfile as sf
import pytest
from music_re.core.analyzer import analyze_audio
from music_re.schemas import TrackAnalysis

def test_analyze_audio_synthetic(tmp_path: Path):
    sr = 22050
    # 120 BPM clicks (every 0.5 sec) over 4 seconds
    duration = 4.0
    audio = np.zeros(int(sr * duration))
    for beat in range(8):
        idx = int(beat * 0.5 * sr)
        audio[idx:idx + 100] = 0.9

    wav_file = tmp_path / "clicks.wav"
    sf.write(str(wav_file), audio, sr)

    result = analyze_audio(wav_file)
    assert isinstance(result, TrackAnalysis)
    assert result.duration_sec >= 3.9
    assert result.bpm > 0
    assert len(result.beat_times) > 0
    assert isinstance(result.key, str)
    assert isinstance(result.lufs, float)

def test_analyze_audio_file_not_found(tmp_path: Path):
    missing_file = tmp_path / "nonexistent.wav"
    with pytest.raises(FileNotFoundError):
        analyze_audio(missing_file)
