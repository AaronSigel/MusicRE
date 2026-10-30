"""Bar-by-bar chord progression detection using chroma template matching."""

from typing import Any, Dict, List, Tuple, Union
import warnings

import librosa
import numpy as np

PITCH_CLASSES: List[str] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]


def build_chord_templates() -> Dict[str, np.ndarray]:
    """Build normalized 12-bin chroma templates for 24 major and minor triads."""
    templates: Dict[str, np.ndarray] = {}
    for i, root in enumerate(PITCH_CLASSES):
        # Major triad: root, major 3rd (+4), perfect 5th (+7)
        maj = np.zeros(12, dtype=np.float32)
        maj[i] = 1.0
        maj[(i + 4) % 12] = 1.0
        maj[(i + 7) % 12] = 1.0
        norm_maj = np.linalg.norm(maj)
        templates[root] = maj / norm_maj if norm_maj > 0 else maj

        # Minor triad: root, minor 3rd (+3), perfect 5th (+7)
        minor = np.zeros(12, dtype=np.float32)
        minor[i] = 1.0
        minor[(i + 3) % 12] = 1.0
        minor[(i + 7) % 12] = 1.0
        norm_min = np.linalg.norm(minor)
        templates[f"{root}m"] = minor / norm_min if norm_min > 0 else minor

    return templates


CHORD_TEMPLATES = build_chord_templates()


def _extract_bar_bounds(bar: Any) -> Tuple[float, float]:
    """Extract (start_sec, end_sec) from dict, tuple/list, or object."""
    if isinstance(bar, dict):
        start = float(bar.get("start_sec", bar.get("start", 0.0)))
        end = float(bar.get("end_sec", bar.get("end", 0.0)))
    elif isinstance(bar, (tuple, list)) and len(bar) >= 2:
        start = float(bar[0])
        end = float(bar[1])
    else:
        start = float(getattr(bar, "start_sec", getattr(bar, "start", 0.0)))
        end = float(getattr(bar, "end_sec", getattr(bar, "end", 0.0)))
    return start, end


def detect_chords_for_bars(
    y: np.ndarray,
    sr: int,
    bars: Union[List[dict], List[Tuple[float, float]], List[Any]],
) -> List[Dict[str, Any]]:
    """Detect chord progression bar-by-bar using chroma template matching.

    Args:
        y: Audio signal array (1D mono or 2D).
        sr: Sample rate in Hz.
        bars: List of bars, where each bar defines start and end timestamps.

    Returns:
        List of dicts containing bar index (1-based), start_sec, end_sec,
        chord label, and confidence score.
    """
    if y is None or sr <= 0 or not bars:
        return []

    # Ensure mono 1D float array
    y_mono = np.asarray(y, dtype=np.float32)
    if y_mono.ndim > 1:
        if y_mono.shape[0] <= 8 and y_mono.shape[1] > y_mono.shape[0]:
            y_mono = np.mean(y_mono, axis=0)
        else:
            y_mono = np.mean(y_mono, axis=1)

    results: List[Dict[str, Any]] = []

    for i, bar in enumerate(bars):
        start_sec, end_sec = _extract_bar_bounds(bar)
        bar_idx = i + 1

        start_samp = max(0, int(round(start_sec * sr)))
        end_samp = min(len(y_mono), int(round(end_sec * sr)))

        if end_samp <= start_samp:
            results.append({
                "bar": bar_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "chord": "N/A",
                "confidence": 0.0,
            })
            continue

        y_slice = y_mono[start_samp:end_samp]

        # Check silence
        peak = float(np.max(np.abs(y_slice)))
        rms = float(np.sqrt(np.mean(y_slice ** 2)))
        if peak < 1e-4 or rms < 1e-4:
            results.append({
                "bar": bar_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "chord": "N/A",
                "confidence": 0.0,
            })
            continue

        # Extract chroma
        chroma = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chroma = librosa.feature.chroma_cqt(y=y_slice, sr=sr)
        except Exception:
            pass

        if chroma is None or chroma.size == 0:
            try:
                n_fft = min(2048, max(256, int(2 ** np.floor(np.log2(len(y_slice))))))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    chroma = librosa.feature.chroma_stft(y=y_slice, sr=sr, n_fft=n_fft)
            except Exception:
                chroma = None

        if chroma is None or chroma.size == 0:
            results.append({
                "bar": bar_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "chord": "N/A",
                "confidence": 0.0,
            })
            continue

        mean_chroma = np.mean(chroma, axis=1)
        norm = float(np.linalg.norm(mean_chroma))
        if norm < 1e-6:
            results.append({
                "bar": bar_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "chord": "N/A",
                "confidence": 0.0,
            })
            continue

        mean_chroma = mean_chroma / norm

        best_chord = "N/A"
        best_sim = -1.0
        for chord_name, template in CHORD_TEMPLATES.items():
            sim = float(np.dot(mean_chroma, template))
            if sim > best_sim:
                best_sim = sim
                best_chord = chord_name

        conf = round(max(0.0, float(best_sim)), 2)
        results.append({
            "bar": bar_idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "chord": best_chord,
            "confidence": conf,
        })

    return results
