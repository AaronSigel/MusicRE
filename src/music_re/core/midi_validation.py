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
    if end_sec <= start_sec:
        return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec,
                            velocity=velocity, reason='invalid_duration', **zero)
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
    # Compare expected energy with total energy in each representation.
    norm = np.max(total)
    if norm <= 1e-10:
        return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec,
                            velocity=velocity, reason=reason, **zero)
    if cqt_in_range:
        cache = getattr(features, '_note_validation_cache', None)
        if cache is None:
            cache = {}
            setattr(features, '_note_validation_cache', cache)
        if pitch not in cache:
            pitch_track = np.mean(features.cqt_mag[cqt_idx], axis=0)
            cache[pitch] = (pitch_track, float(np.max(pitch_track)) * 0.08)
        pitch_track, persistence_threshold = cache[pitch]
        persistence = float(np.mean(pitch_track[frames] > persistence_threshold))
    else:
        persistence_threshold = float(np.max(fundamental)) * .08
        persistence = float(np.mean(fundamental > persistence_threshold))
    window_rms = features.rms[frames] + 1e-12
    fundamental_relative = np.clip(cqt / (25.0 * window_rms), 0.0, 1.0)
    support_frames = (0.2 * fundamental_relative + 0.8 * np.clip(harmonic_presence, 0, 1)
                      if cqt_in_range else np.clip(harmonic_presence, 0, 1))
    support = float(np.mean(support_frames))
    background_levels = []
    for k in range(1, 5):
        target = f0 * k
        distance = np.abs(freqs - target)
        local = (freqs >= target * 2 ** (-3 / 12)) & (freqs <= target * 2 ** (3 / 12))
        harmonic_bin = distance <= max(12, target * .018)
        local &= ~harmonic_bin
        if local.any():
            background_levels.append(np.mean(mag[local], axis=0))
    background = np.mean(background_levels, axis=0) if background_levels else np.zeros(len(frames))
    contrast_energy = harmonic if cqt_in_range else fundamental
    contrast = np.mean(contrast_energy / (contrast_energy + background + 1e-12))
    neighbor_energy = []
    for p in (pitch - 12, pitch - 1, pitch + 1, pitch + 12):
        idx = np.flatnonzero(np.isclose(features.cqt_midi, p, atol=.51))
        neighbor_energy.append(float(np.mean(features.cqt_mag[idx][:, frames])) if idx.size else 0.0)
    expected = float(np.mean(cqt))
    leakage = max((v - expected) / (v + expected + 1e-12) for v in neighbor_energy)
    leakage = max(0.0, leakage)
    flux = features.flux
    onset_idx = int(np.argmin(np.abs(features.times - start_sec)))
    window = 3
    near_mask = np.zeros(len(flux), dtype=bool)
    near_mask[max(0, onset_idx-window):min(len(flux), onset_idx+window+1)] = True
    near = flux[near_mask]
    elsewhere = features.flux[frames]
    if len(elsewhere):
        elsewhere_times = np.arange(len(features.times))[frames]
        elsewhere = flux[frames][np.abs(elsewhere_times - onset_idx) > window]
    onset = float(np.max(near)) / (float(np.max(elsewhere)) + 1e-12) if near.size and elsewhere.size else (1.0 if near.size else 0.0)
    return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec, velocity=velocity,
                        harmonic_support=_bounded(support), spectral_contrast=_bounded(contrast),
                        temporal_persistence=_bounded(persistence), onset_agreement=_bounded(onset),
                        leakage=_bounded(leakage), reason=reason)
