"""Per-note evidence extraction from precomputed stem spectral features."""
from __future__ import annotations

import numpy as np

from music_re.core.spectral import StemSpectralFeatures, midi_to_hz
from music_re.schemas import NoteEvidence


def _bounded(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0


def compute_note_evidence(
    features: StemSpectralFeatures,
    pitch: int,
    start_sec: float,
    end_sec: float,
    velocity: int = 100,
) -> NoteEvidence:
    """Measure pitch evidence in a note window; scores are descriptive, not probabilities."""
    n = len(features.times)
    zero = dict(harmonic_support=0.0, spectral_contrast=0.0,
                temporal_persistence=0.0, onset_agreement=0.0, leakage=0.0)
    reason = None
    if n == 0 or end_sec <= 0 or start_sec >= (features.times[-1] + features.hop_length / features.sr):
        reason = 'out_of_range'
        return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec,
                            velocity=velocity, reason=reason, **zero)
    hop = features.hop_length / features.sr
    lo = max(0, int(np.floor(start_sec / hop)))
    hi = min(n, int(np.ceil(end_sec / hop)))
    if hi <= lo:
        hi = min(n, lo + 1)
    if hi <= lo:
        return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec,
                            velocity=velocity, reason='out_of_range', **zero)
    frames = np.arange(lo, hi)
    cqt_idx = np.flatnonzero(np.isclose(features.cqt_midi, pitch, atol=0.51))
    cqt_in_range = bool(cqt_idx.size)
    if not cqt_in_range:
        reason = 'pitch_outside_cqt_range_stft_only'
    f0 = midi_to_hz(pitch)
    mag = features.stft_mag[:, frames]
    freqs = features.stft_freqs
    # Integrate narrow fundamental/harmonic bands, then normalize by window spectral energy.
    harmonic = np.zeros(len(frames), dtype=np.float64)
    fundamental = np.zeros(len(frames), dtype=np.float64)
    harmonic_presence = np.zeros(len(frames), dtype=np.float64)
    for k in range(1, 5):
        mask = np.abs(freqs - f0 * k) <= max(12.0, f0 * k * 0.018)
        if mask.any():
            band = np.max(mag[mask], axis=0)
            harmonic += band / k
            nearby = np.abs(freqs - f0 * k)
            background_mask = (nearby >= max(25.0, f0 * k * 0.03)) & (nearby <= max(80.0, f0 * k * 0.12))
            background_level = np.mean(mag[background_mask], axis=0) if background_mask.any() else np.zeros(len(frames))
            harmonic_presence += band / (band + 8.0 * background_level + 1e-12)
            if k == 1:
                fundamental = band
    harmonic_presence /= 4.0
    cqt = np.mean(features.cqt_mag[cqt_idx][:, frames], axis=0) if cqt_in_range else fundamental
    pitch_energy = np.maximum(cqt, fundamental)
    total = np.sum(mag, axis=0) + 1e-12
    # Normalize pitch energy against all available CQT pitches, per frame.
    cqt_frame_max = np.max(features.cqt_mag[:, frames], axis=0) if features.cqt_mag.shape[0] else np.zeros(len(frames))
    pitch_relative = cqt / (cqt_frame_max + 1e-12)
    norm = np.max(total)
    if norm <= 1e-10:
        return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec,
                            velocity=velocity, reason=reason, **zero)
    if cqt_in_range:
        pitch_track = np.mean(features.cqt_mag[cqt_idx], axis=0)
        persistence_threshold = float(np.max(pitch_track)) * 0.08
        persistence = float(np.mean(pitch_track[frames] > persistence_threshold))
    else:
        persistence_threshold = float(np.max(fundamental)) * .08
        persistence = float(np.mean(fundamental > persistence_threshold))
    harmonic_relative = np.clip(harmonic_presence, 0.0, 1.0)
    support_frames = (0.7 * np.clip(pitch_relative, 0, 1) + 0.3 * harmonic_relative
                      if cqt_in_range else harmonic_relative)
    support = float(np.mean(support_frames))
    background_mask = np.ones(len(freqs), dtype=bool)
    for k in range(1, 5):
        background_mask &= np.abs(freqs - f0 * k) > max(25, f0 * k * .025)
    background = np.mean(mag[background_mask], axis=0) if background_mask.any() else np.zeros(len(frames))
    contrast = np.mean(pitch_energy / (pitch_energy + background + 1e-12))
    neighbor_energy = []
    for p in (pitch - 12, pitch - 1, pitch + 1, pitch + 12):
        idx = np.flatnonzero(np.isclose(features.cqt_midi, p, atol=.51))
        neighbor_energy.append(float(np.mean(features.cqt_mag[idx][:, frames])) if idx.size else 0.0)
    expected = float(np.mean(cqt))
    leakage = max((v - expected) / (v + expected + 1e-12) for v in neighbor_energy)
    leakage = max(0.0, leakage)
    flux = features.flux[frames]
    onset_idx = int(np.argmin(np.abs(features.times[frames] - start_sec)))
    window = max(1, int(round(3 * hop / hop)))
    near = flux[max(0, onset_idx-window):min(len(flux), onset_idx+window+1)]
    onset = float(np.max(near)) / (float(np.max(flux)) + 1e-12) if flux.size else 0.0
    return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec, velocity=velocity,
                        harmonic_support=_bounded(support), spectral_contrast=_bounded(contrast),
                        temporal_persistence=_bounded(persistence), onset_agreement=_bounded(onset),
                        leakage=_bounded(leakage), reason=reason)
