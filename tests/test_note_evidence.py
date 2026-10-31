import numpy as np
import pytest

from music_re.core.spectral import SpectralConfig, compute_stem_features
from music_re.core.midi_validation import compute_note_evidence

FEATURES = ('harmonic_support', 'spectral_contrast', 'temporal_persistence', 'onset_agreement', 'leakage')


def tone(freqs, duration=1.0, gain=0.2, sr=22050):
    t = np.arange(int(sr * duration)) / sr
    return (gain * sum(np.sin(2 * np.pi * f * t) for f in freqs)).astype(np.float32)


def ev(y, pitch, start=0.1, end=0.8):
    return compute_note_evidence(compute_stem_features(y, SpectralConfig()), pitch, start, end)


@pytest.mark.parametrize('pitch', [69, 60])
def test_correct_and_wrong_pitch_evidence_is_finite(pitch):
    note = ev(tone([440]), pitch)
    assert note.harmonic_support > 0
    assert all(np.isfinite(getattr(note, field)) for field in FEATURES)


def test_correct_pitch_beats_wrong_pitch():
    correct, wrong = ev(tone([440]), 69), ev(tone([440]), 60)
    assert correct.harmonic_support > 0.01
    assert correct.harmonic_support > wrong.harmonic_support
    assert correct.temporal_persistence > wrong.temporal_persistence


def test_octave_error_has_higher_leakage():
    correct, octave = ev(tone([440]), 69), ev(tone([440]), 81)
    assert octave.leakage > 0.01
    assert octave.leakage > correct.leakage


def test_polyphonic_notes_both_have_support():
    mixture = tone([440, 523.25])
    assert ev(mixture, 69).harmonic_support > 0.01
    assert ev(mixture, 72).harmonic_support > 0.01


def test_bass_55_hz_beats_58_27_hz_neighbor():
    bass, neighbor = ev(tone([55]), 33), ev(tone([58.27]), 33)
    assert bass.harmonic_support > 0.01
    assert bass.harmonic_support > neighbor.harmonic_support


def test_quiet_minus_40_db_preserves_feature_ordering():
    loud = ev(tone([440]), 69)
    quiet = ev(tone([440], gain=0.002), 69)
    assert quiet.harmonic_support == pytest.approx(loud.harmonic_support, abs=0.02)
    assert quiet.temporal_persistence == loud.temporal_persistence


def test_vibrato_within_30_cents_remains_supported():
    t = np.arange(22050) / 22050
    instantaneous = 440 * 2 ** ((30 / 100) * np.sin(2 * np.pi * 5 * t) / 12)
    phase = 2 * np.pi * np.cumsum(instantaneous) / 22050
    assert ev((0.2 * np.sin(phase)).astype(np.float32), 69).temporal_persistence > 0.8


def test_decaying_reverb_tail_has_lower_persistence():
    sr = 22050
    t = np.arange(sr) / sr
    y = np.zeros(sr, np.float32)
    sustain = (t < 0.4)
    y[sustain] = (0.2 * np.sin(2 * np.pi * 440 * t[sustain])).astype(np.float32)
    tail = t >= 0.4
    y[tail] = (0.2 * np.exp(-(t[tail] - 0.4) * 14) * np.sin(2 * np.pi * 440 * t[tail])).astype(np.float32)
    sustained_note, tail_note = ev(y, 69, 0.1, 0.35), ev(y, 69, 0.55, 0.95)
    assert tail_note.temporal_persistence < sustained_note.temporal_persistence


def test_remote_tone_does_not_change_local_spectral_contrast():
    base = tone([440])
    remote = base + tone([4000], gain=20.0)
    assert ev(base, 69).spectral_contrast == pytest.approx(ev(remote, 69).spectral_contrast, abs=0.005)


def test_short_zero_flux_note_has_zero_onset_agreement():
    features = compute_stem_features(tone([440]), SpectralConfig())
    features.flux[:] = 0
    note = compute_note_evidence(features, 69, 0.2, 0.24)
    assert note.onset_agreement == 0


def test_onset_peak_two_hops_before_note_start_is_included():
    sr = 22050
    hop = 256 / sr
    start = 0.3
    onset_time = start - 2 * hop
    t = np.arange(sr) / sr
    y = np.zeros(sr, np.float32)
    active = t >= onset_time
    y[active] = (0.2 * np.sin(2 * np.pi * 440 * t[active])).astype(np.float32)
    features = compute_stem_features(y, SpectralConfig())
    features.flux[:] = 0
    onset_frame = int(round(onset_time / hop))
    features.flux[onset_frame] = 1
    note = compute_note_evidence(features, 69, start, 0.8)
    assert note.harmonic_support > 0.01
    assert note.onset_agreement == pytest.approx(1.0)


@pytest.mark.parametrize(('start', 'end'), [(0.2, 0.2), (0.3, 0.2)])
def test_invalid_duration_returns_zero_evidence(start, end):
    note = ev(tone([440]), 69, start, end)
    assert note.reason == 'invalid_duration'
    assert all(getattr(note, field) == 0 for field in FEATURES)


def test_broadband_onset_click_does_not_support_wrong_pitch():
    base = tone([440])
    clicked = base.copy()
    clicked[int(0.1 * 22050)] += 1.0
    assert ev(clicked, 60).harmonic_support < ev(base, 69).harmonic_support


def test_note_past_audio_end_is_clipped_and_finite():
    note = ev(tone([440], duration=0.5), 69, 0.3, 1.0)
    assert note.harmonic_support > 0
    assert all(np.isfinite(getattr(note, field)) for field in FEATURES)


def test_silent_stem_returns_zero_features():
    note = ev(np.zeros(22050, np.float32), 69)
    assert all(getattr(note, field) == 0 for field in FEATURES)


def test_fully_out_of_range_note_has_reason_and_zero_features():
    note = ev(tone([440]), 69, 3, 4)
    assert note.reason == 'out_of_range'
    assert all(getattr(note, field) == 0 for field in FEATURES)


def test_sub_hop_notes_use_at_least_one_frame():
    start, end = 0.2, 0.205
    note = ev(tone([440]), 69, start, end)
    assert note.harmonic_support > 0


def test_harmonic_tone_adds_support_without_hurting_fundamental():
    pure = ev(tone([440]), 69)
    harmonic = ev(tone([440, 880, 1320]), 69)
    wrong = ev(tone([440, 880, 1320]), 60)
    assert harmonic.harmonic_support >= pure.harmonic_support
    assert harmonic.harmonic_support > wrong.harmonic_support


@pytest.mark.parametrize('signal,pitch,start,end', [
    (tone([440]), 69, 0.1, 0.8),
    (tone([440]), 60, 0.1, 0.8),
    (tone([440, 880, 1320]), 69, 0.1, 0.8),
    (tone([55]), 33, 0.1, 0.8),
    (tone([58.27]), 33, 0.1, 0.8),
    (np.zeros(22050, np.float32), 69, 0.1, 0.8),
    (tone([440], duration=.5), 69, .3, 1.0),
    (tone([440]), 69, 3, 4),
    (tone([440]), 69, .2, .2),
    (tone([440]), 69, .2, .205),
])
def test_all_features_are_finite_and_bounded(signal, pitch, start, end):
    note = ev(signal, pitch, start, end)
    assert all(np.isfinite(getattr(note, field)) and 0 <= getattr(note, field) <= 1 for field in FEATURES)
