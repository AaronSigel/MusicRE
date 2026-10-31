import numpy as np

from music_re.core.spectral import SpectralConfig, compute_stem_features
from music_re.core.midi_validation import compute_note_evidence


def tone(freqs, duration=1.0, gain=0.2, sr=22050):
    t = np.arange(int(sr * duration)) / sr
    return (gain * sum(np.sin(2 * np.pi * f * t) for f in freqs)).astype(np.float32)


def evidence(y, pitch, start=0.1, end=0.8):
    features = compute_stem_features(y, SpectralConfig())
    return compute_note_evidence(features, pitch, start, end)


def test_correct_pitch_beats_wrong_pitch_and_octave_leakage():
    y = tone([440])
    a4, c4, a5 = evidence(y, 69), evidence(y, 60), evidence(y, 81)
    assert a4.harmonic_support > c4.harmonic_support
    assert a4.temporal_persistence > c4.temporal_persistence
    assert a5.leakage > a4.leakage


def test_polyphony_bass_vibrato_and_quiet_gain():
    both = tone([440, 523.25])
    assert evidence(both, 69).harmonic_support > 0
    assert evidence(both, 72).harmonic_support > 0
    bass = evidence(tone([55]), 33)
    neighbour = evidence(tone([58.27]), 33)
    assert bass.harmonic_support > neighbour.harmonic_support
    t = np.arange(22050) / 22050
    vibrato = (0.2 * np.sin(2 * np.pi * 440 * 2 ** ((30 / 100) * np.sin(2*np.pi*5*t) / 12) * t)).astype(np.float32)
    assert evidence(vibrato, 69).temporal_persistence > 0
    loud, quiet = evidence(tone([440]), 69), evidence(tone([440], gain=0.002), 69)
    assert abs(loud.harmonic_support - quiet.harmonic_support) < 0.15


def test_edge_cases_are_finite_bounded_and_silent_is_zero():
    silent = evidence(np.zeros(22050, np.float32), 69)
    assert all(getattr(silent, name) == 0 for name in (
        'harmonic_support', 'spectral_contrast', 'temporal_persistence', 'onset_agreement', 'leakage'))
    outside = evidence(tone([440]), 69, 3, 4)
    assert outside.reason == 'out_of_range'
    past_end = evidence(tone([440], duration=0.5), 69, 0.3, 1.0)
    for item in (silent, outside, past_end):
        for name in ('harmonic_support', 'spectral_contrast', 'temporal_persistence', 'onset_agreement', 'leakage'):
            value = getattr(item, name)
            assert np.isfinite(value) and 0 <= value <= 1
