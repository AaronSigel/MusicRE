from pathlib import Path
from typing import List, Tuple, Union
import librosa
import numpy as np
import pretty_midi
import scipy.signal

from music_re.core.audio_io import decode_audio


def _bandpass_filter(
    y: np.ndarray, sr: int, low_cut: float, high_cut: float, order: int = 3
) -> np.ndarray:
    """Butterworth bandpass filter."""
    if len(y) <= 21:
        return y

    nyquist = 0.5 * sr
    low = max(0.001, low_cut / nyquist)
    high = min(0.999, high_cut / nyquist)
    if low >= high:
        return y
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    return scipy.signal.filtfilt(b, a, y)


def _detect_band_onsets(
    filtered_audio: np.ndarray,
    sr: int,
    pitch: int,
    delta: float = 0.08,
) -> List[Tuple[float, int, int]]:
    """Detect onsets in filtered audio band and estimate RMS velocity."""
    if len(filtered_audio) <= 21 or np.max(np.abs(filtered_audio)) < 1e-4:
        return []

    # Onset detection using spectral flux
    onset_frames = librosa.onset.onset_detect(
        y=filtered_audio,
        sr=sr,
        units="frames",
        delta=delta,
        backtrack=False,
    )

    if len(onset_frames) == 0:
        return []

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    events = []

    # Calculate local RMS around each onset for velocity
    hop_length = 512
    rms = librosa.feature.rms(y=filtered_audio, hop_length=hop_length)[0]
    max_rms = np.max(rms) if len(rms) > 0 and np.max(rms) > 0 else 1.0

    for frame, onset_time in zip(onset_frames, onset_times):
        frame_idx = min(frame, len(rms) - 1)
        local_rms = rms[frame_idx]
        norm_val = min(1.0, max(0.1, local_rms / max_rms))
        velocity = int(30 + norm_val * 97)
        events.append((float(onset_time), pitch, velocity))

    return events


def transcribe_drums(
    audio_path: Union[str, Path],
    output_midi_path: Union[str, Path],
    bpm: float = 120.0,
) -> Path:
    """Transcribe drum stem to GM MIDI drum track using multi-band onset detection."""
    in_p = Path(audio_path).resolve()
    out_p = Path(output_midi_path).resolve()

    if not in_p.is_file():
        raise FileNotFoundError(f"Drum audio file not found: {in_p}")

    out_p.parent.mkdir(parents=True, exist_ok=True)

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    drum_instrument = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

    sr = 22050
    y, sr = decode_audio(in_p, target_sr=sr, normalize_db=None)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if len(y) <= 21 or np.max(np.abs(y)) < 1e-5:
        pm.instruments.append(drum_instrument)
        pm.write(str(out_p))
        return out_p

    # Multi-band frequency definitions:
    # 1. Kick (MIDI 36): 30 Hz - 140 Hz
    # 2. Snare (MIDI 38): 200 Hz - 900 Hz
    # 3. Hi-Hat (MIDI 42): 4500 Hz - 10000 Hz
    bands = [
        ("kick", 36, 30.0, 140.0, 0.05),
        ("snare", 38, 220.0, 900.0, 0.07),
        ("hihat", 42, 4500.0, 10000.0, 0.04),
    ]

    all_events: List[Tuple[float, int, int]] = []
    for name, pitch, low_cut, high_cut, delta in bands:
        filtered = _bandpass_filter(y, sr, low_cut, high_cut)
        events = _detect_band_onsets(filtered, sr, pitch, delta=delta)
        all_events.extend(events)

    # Sort events by time
    all_events.sort(key=lambda x: x[0])

    # Quantize to 16th note grid
    grid_sec = (60.0 / bpm) * 0.25
    hit_duration = 0.08

    seen_hits = set()
    for onset_sec, pitch, velocity in all_events:
        snapped_start = round(onset_sec / grid_sec) * grid_sec
        key = (snapped_start, pitch)
        if key in seen_hits:
            continue
        seen_hits.add(key)

        drum_instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=snapped_start,
                end=snapped_start + hit_duration,
            )
        )

    pm.instruments.append(drum_instrument)
    pm.write(str(out_p))
    return out_p
