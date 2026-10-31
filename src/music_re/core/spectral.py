"""Pure-DSP spectral feature engine — no trained models, no file I/O except load_mono."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import scipy.signal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralConfig:
    sr: int = 22050
    n_fft: int = 4096
    hop_length: int = 256
    window: str = 'hann'
    cqt_bins: int = 84
    cqt_bins_per_octave: int = 12
    cqt_fmin_midi: int = 24   # C1
    chunk_sec: float = 60.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class StemSpectralFeatures:
    sr: int
    hop_length: int
    times: np.ndarray           # frame-centre times, s   [frames]
    stft_freqs: np.ndarray      # Hz per STFT bin          [freq]
    stft_mag: np.ndarray        # float32 [freq, frames]
    cqt_mag: np.ndarray         # float32 [bins, frames]
    cqt_midi: np.ndarray        # MIDI pitch per CQT bin   [bins]
    rms: np.ndarray             # per frame                [frames]
    flux: np.ndarray            # per frame                [frames]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def midi_to_hz(pitch: float) -> float:
    """Convert MIDI pitch number to frequency in Hz (A4=69=440 Hz)."""
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


def _to_mono_float32(y: np.ndarray) -> np.ndarray:
    """Downmix to mono float32.  Accepts [samples] or [channels, samples]."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=0)
    elif y.ndim > 2:
        y = np.mean(y.reshape(-1, y.shape[-1]), axis=0)
    return y


def _pad_for_fft(y: np.ndarray, n_fft: int) -> np.ndarray:
    """Zero-pad short signals so librosa never emits 'n_fft too large' warnings."""
    if y.shape[-1] < n_fft:
        pad_width = n_fft - y.shape[-1]
        y = np.pad(y, (0, pad_width))
    return y


# ---------------------------------------------------------------------------
# Individual DSP functions
# ---------------------------------------------------------------------------

def compute_stft(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 4096,
    hop_length: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute magnitude STFT.

    Returns:
        freqs: Hz per bin [freq]
        times: frame-centre times in s [frames]
        mag:   float32 magnitude [freq, frames]
    """
    y = _to_mono_float32(y)
    if y.size == 0:
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return freqs, np.zeros(0, dtype=np.float32), np.zeros((len(freqs), 0), dtype=np.float32)

    return _compute_stft(y, sr, n_fft, hop_length, 'hann')


def _compute_stft(
    y: np.ndarray, sr: int, n_fft: int, hop_length: int, window: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Internal STFT allowing composite extraction to honor configured windows."""
    if y.size == 0:
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
        return freqs, np.zeros(0, dtype=np.float32), np.zeros((len(freqs), 0), dtype=np.float32)
    y = _pad_for_fft(y, n_fft)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, center=True)
    mag = np.abs(stft)
    del stft
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
    times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length).astype(np.float32)
    return freqs, times, mag


def compute_cqt(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin_midi: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CQT magnitude.

    Returns:
        cqt_mag:  float32 [bins, frames]
        cqt_midi: MIDI pitch per bin [bins]
    """
    fmin = midi_to_hz(fmin_midi)
    cqt_midi = np.array(
        [fmin_midi + b * 12.0 / bins_per_octave for b in range(n_bins)],
        dtype=np.float32,
    )

    y = _to_mono_float32(y)
    if y.size == 0:
        return np.zeros((n_bins, 0), dtype=np.float32), cqt_midi

    # The lowest CQT octave (fmin≈32.7 Hz, 84 bins/12bpo) requires librosa's
    # internal STFT n_fft=256 on a block length of at least 128 samples;
    # empirically ≥16384 samples silences the 'n_fft too large' warning.
    min_len = 16384
    if y.size < min_len:
        y = np.pad(y, (0, min_len - y.size))

    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
    )
    mag = np.abs(cqt)
    del cqt
    return mag, cqt_midi


def compute_rms(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 256,
) -> np.ndarray:
    """Per-frame RMS energy, float32."""
    y = _to_mono_float32(y)
    if y.size == 0:
        return np.zeros(0, dtype=np.float32)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    return rms[0].astype(np.float32)


def compute_spectral_flux(mag: np.ndarray) -> np.ndarray:
    """Half-wave-rectified spectral flux, same frame count as mag.

    Args:
        mag: [freq, frames] float magnitude array.

    Returns:
        flux: float32 [frames]
    """
    mag = np.asarray(mag, dtype=np.float32)
    n_frames = mag.shape[1] if mag.ndim == 2 else 0
    if n_frames == 0:
        return np.zeros(n_frames, dtype=np.float32)

    diff = np.diff(mag, prepend=mag[:, :1], axis=1)   # [freq, frames]
    np.maximum(diff, 0.0, out=diff)
    flux = diff.sum(axis=0, dtype=np.float32)
    return flux


def detect_harmonic_peaks(
    mag_column: np.ndarray,
    freqs: np.ndarray,
    f0: float,
    num_harmonics: int = 4,
    tolerance_hz: float = 15.0,
) -> list[float]:
    """Find spectral peaks near the first `num_harmonics` harmonics of f0.

    Args:
        mag_column: 1-D magnitude array (one frame or mean across frames).
        freqs:      corresponding frequency array in Hz.
        f0:         fundamental frequency in Hz.
        num_harmonics: how many harmonics to search for.
        tolerance_hz:  search window around each harmonic.

    Returns:
        List of detected peak frequencies (Hz) as floats.
    """
    mag_column = np.asarray(mag_column, dtype=np.float32)
    freqs = np.asarray(freqs, dtype=np.float32)

    if mag_column.size == 0 or freqs.size == 0:
        return []

    peaks_idx, _ = scipy.signal.find_peaks(mag_column)
    if peaks_idx.size == 0:
        return []

    found: list[float] = []
    for k in range(1, num_harmonics + 1):
        target = f0 * k
        mask = np.abs(freqs[peaks_idx] - target) <= tolerance_hz
        candidates = peaks_idx[mask]
        if candidates.size > 0:
            best = candidates[np.argmax(mag_column[candidates])]
            found.append(float(freqs[best]))
    return found


# ---------------------------------------------------------------------------
# Composite feature extractor
# ---------------------------------------------------------------------------

def compute_stem_features(y: np.ndarray, config: SpectralConfig) -> StemSpectralFeatures:
    """One STFT+CQT+RMS+flux pass over a mono (or stereo) signal.

    All frame axes are aligned to the same hop_length and the same frame count.
    Stereo input is downmixed to mono before processing.

    Args:
        y:      Audio signal [samples] or [channels, samples], float32.
        config: SpectralConfig with DSP parameters.

    Returns:
        StemSpectralFeatures with all arrays aligned to the same frame axis.
    """
    y_mono = _to_mono_float32(y)

    if config.chunk_sec > 0 and y_mono.size > config.chunk_sec * config.sr:
        return _compute_chunked_features(y_mono, config)

    # STFT
    stft_freqs, times, stft_mag = _compute_stft(
        y_mono, sr=config.sr, n_fft=config.n_fft, hop_length=config.hop_length,
        window=config.window,
    )
    n_frames = 1 + y_mono.size // config.hop_length if y_mono.size else 0
    stft_mag = stft_mag[:, :n_frames]
    times = times[:n_frames]

    # CQT — trim or pad to match STFT frame count
    cqt_mag, cqt_midi = compute_cqt(
        y_mono,
        sr=config.sr,
        hop_length=config.hop_length,
        n_bins=config.cqt_bins,
        bins_per_octave=config.cqt_bins_per_octave,
        fmin_midi=config.cqt_fmin_midi,
    )
    if cqt_mag.shape[1] > n_frames:
        cqt_mag = cqt_mag[:, :n_frames]
    elif cqt_mag.shape[1] < n_frames:
        pad = n_frames - cqt_mag.shape[1]
        cqt_mag = np.pad(cqt_mag, ((0, 0), (0, pad)))

    # RMS — trim or pad to match
    rms = compute_rms(y_mono, frame_length=config.n_fft, hop_length=config.hop_length)
    if rms.shape[0] > n_frames:
        rms = rms[:n_frames]
    elif rms.shape[0] < n_frames:
        rms = np.pad(rms, (0, n_frames - rms.shape[0]))

    # A shorter centered window limits onset smearing while preserving frame grid.
    flux_n_fft = min(1024, config.n_fft)
    _, _, flux_mag = _compute_stft(
        y_mono, sr=config.sr, n_fft=flux_n_fft, hop_length=config.hop_length,
        window=config.window,
    )
    flux = compute_spectral_flux(flux_mag[:, :n_frames])
    del flux_mag

    return StemSpectralFeatures(
        sr=config.sr,
        hop_length=config.hop_length,
        times=times,
        stft_freqs=stft_freqs,
        stft_mag=stft_mag,
        cqt_mag=cqt_mag,
        cqt_midi=cqt_midi,
        rms=rms,
        flux=flux,
    )


def _compute_chunked_features(y: np.ndarray, config: SpectralConfig) -> StemSpectralFeatures:
    hop = config.hop_length
    octaves = int(np.ceil(config.cqt_bins / config.cqt_bins_per_octave))
    alignment = 2 ** max(0, octaves - 1)
    chunk_frames = max(alignment, int(config.chunk_sec * config.sr / hop) // alignment * alignment)
    q = 1.0 / (2 ** (1.0 / config.cqt_bins_per_octave) - 1.0)
    filter_len = q * config.sr / midi_to_hz(config.cqt_fmin_midi)
    overlap = int(np.ceil(2 * max(config.n_fft, filter_len) / hop / alignment)) * alignment
    frames = 1 + len(y) // hop
    whole_config = dataclasses.replace(config, chunk_sec=0)
    result = None
    for start in range(0, frames, chunk_frames):
        end = min(start + chunk_frames, frames)
        padded_start = max(0, start - overlap)
        padded_end = min(len(y), (end + overlap) * hop)
        part = compute_stem_features(y[padded_start * hop:padded_end], whole_config)
        if result is None:
            result = StemSpectralFeatures(
                sr=config.sr, hop_length=hop,
                times=(np.arange(frames) * (hop / config.sr)).astype(np.float32),
                stft_freqs=part.stft_freqs,
                stft_mag=np.empty((part.stft_mag.shape[0], frames), dtype=np.float32),
                cqt_mag=np.empty((part.cqt_mag.shape[0], frames), dtype=np.float32),
                cqt_midi=part.cqt_midi,
                rms=np.empty(frames, dtype=np.float32),
                flux=np.empty(frames, dtype=np.float32),
            )
        selection = slice(start - padded_start, end - padded_start)
        result.stft_mag[:, start:end] = part.stft_mag[:, selection]
        result.cqt_mag[:, start:end] = part.cqt_mag[:, selection]
        result.rms[start:end] = part.rms[selection]
        result.flux[start:end] = part.flux[selection]
        del part
    return result


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------

def load_mono(path: Union[str, Path], sr: int = 22050) -> np.ndarray:
    """Decode audio file to mono float32 at the requested sample rate.

    Uses music_re.core.audio_io.decode_audio with normalize_db=None (raw levels).
    Stereo or multichannel input is downmixed by averaging channels, matching
    the same downmix used in pipeline.py.

    Args:
        path: Path to any audio file supported by FFmpeg.
        sr:   Target sample rate.

    Returns:
        Mono float32 numpy array.
    """
    from music_re.core.audio_io import decode_audio

    data, _ = decode_audio(path, target_sr=sr, normalize_db=None)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32)
