from pathlib import Path
from typing import List, Tuple, Union

import librosa
import numpy as np
import scipy.signal


def _bandpass_filter(y: np.ndarray, sr: int, low_cut: float, high_cut: float,
                     order: int = 3) -> np.ndarray:
    """Butterworth bandpass filter."""
    if len(y) <= 21:
        return y
    nyquist = 0.5 * sr
    low, high = max(0.001, low_cut / nyquist), min(0.999, high_cut / nyquist)
    if low >= high:
        return y
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, y)


def _detect_band_onsets(filtered_audio: np.ndarray, sr: int, pitch: int,
                        delta: float = 0.08) -> List[Tuple[float, int, int]]:
    """Detect spectral-flux onsets with local RMS-derived velocity."""
    if len(filtered_audio) <= 21 or np.max(np.abs(filtered_audio)) < 1e-4:
        return []
    frames = librosa.onset.onset_detect(y=filtered_audio, sr=sr, units='frames',
                                       delta=delta, backtrack=False)
    if len(frames) == 0:
        return []
    rms = librosa.feature.rms(y=filtered_audio, hop_length=512)[0]
    max_rms = max(float(np.max(rms)), 1e-12)
    return [(float(t), pitch, int(30 + min(1.0, max(.1, rms[min(f, len(rms)-1)] / max_rms)) * 97))
            for f, t in zip(frames, librosa.frames_to_time(frames, sr=sr))]


def transcribe_drums(audio_path: Union[str, Path], output_midi_path: Union[str, Path],
                     bpm: float = 120.0, grid_subdivisions: int = 4,
                     quantize_ends: bool = True) -> Path:
    """Compatibility wrapper around the canonical drum adapter."""
    from music_re.core.backends.drum_backend import DrumTranscriber
    result = DrumTranscriber().transcribe(audio_path, output_midi_path, bpm=bpm, save_raw=False,
                                          grid_subdivisions=grid_subdivisions,
                                          quantize_ends=quantize_ends)
    return Path(result.midi_path)
