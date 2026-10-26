from pathlib import Path
from typing import Optional, Tuple, Union
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from music_re.core.audio_io import decode_audio
from music_re.schemas import TrackAnalysis

# Pitch class names
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(y: np.ndarray, sr: int) -> Tuple[str, float]:
    """Estimate musical key using Krumhansl-Schmuckler chroma correlation."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)

    # Normalize
    if np.sum(chroma_avg) > 0:
        chroma_norm = chroma_avg / np.sum(chroma_avg)
    else:
        return "C major", 0.0

    best_key = "C major"
    best_corr = -1.0

    for i, root in enumerate(PITCH_CLASSES):
        # Major
        maj_profile_shifted = np.roll(MAJOR_PROFILE, i)
        maj_profile_shifted = maj_profile_shifted / np.sum(maj_profile_shifted)
        maj_corr = float(np.corrcoef(chroma_norm, maj_profile_shifted)[0, 1])
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = f"{root} major"

        # Minor
        min_profile_shifted = np.roll(MINOR_PROFILE, i)
        min_profile_shifted = min_profile_shifted / np.sum(min_profile_shifted)
        min_corr = float(np.corrcoef(chroma_norm, min_profile_shifted)[0, 1])
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = f"{root} minor"

    confidence = max(0.0, min(1.0, (best_corr + 1.0) / 2.0))
    return best_key, round(confidence, 3)


def measure_lufs(y: np.ndarray, sr: int) -> float:
    """Calculate integrated loudness in LUFS using pyloudnorm."""
    # Ensure minimum length for pyloudnorm (block size requires ~400ms)
    min_len = int(sr * 0.45)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))

    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y)
        if np.isneginf(loudness) or np.isnan(loudness):
            return -70.0
        return round(float(loudness), 2)
    except Exception:
        return -70.0


def analyze_audio(file_path: Union[str, Path], stem_name: Optional[str] = None) -> TrackAnalysis:
    """Analyze tempo, key, loudness, and beat grid for audio file."""
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    target_sr = 22050
    # Unified FFmpeg decoding to avoid audioread deprecation warnings
    y, sr = decode_audio(p, target_sr=target_sr, normalize_db=None)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    duration_sec = round(float(len(y) / sr), 2)

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # Convert numpy scalar or 1D array to float
    bpm = round(float(np.atleast_1d(tempo)[0]), 2)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    # Retain full track beat grid
    beat_times = [round(t, 3) for t in beat_times]

    # Key estimation: skip on drums
    is_drums = (stem_name and stem_name.lower() in ("drums", "percussion")) or p.stem.lower() in ("drums", "percussion")
    if is_drums:
        key = "N/A"
        confidence = 0.0
    else:
        key, confidence = estimate_key(y, sr)

    # LUFS calculation
    lufs = measure_lufs(y, sr)

    return TrackAnalysis(
        file_path=str(p),
        duration_sec=duration_sec,
        bpm=bpm,
        key=key,
        key_confidence=confidence,
        lufs=lufs,
        beat_times=beat_times,
    )
