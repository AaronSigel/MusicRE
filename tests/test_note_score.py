import dataclasses
import math

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from music_re.core import midi_validation as mv
from music_re.core.midi_validation import (
    SCORE_FEATURES, RoleRule, SpectralMidiValidator, SpectralValidatorConfig, note_score, score_decision,
)
from music_re.schemas import NoteEvidence

N = len(SCORE_FEATURES)


def _rule(weights=None, bias=0.0, reject=.3, uncertain=.4, **kw):
    return RoleRule(True, .1, .2, .1, .2,
                    score_weights=tuple(weights or [1.0] + [0.0] * (N - 1)), score_bias=bias,
                    score_mean=(0.0,) * N, score_scale=(1.0,) * N,
                    reject_below=reject, uncertain_below=uncertain, **kw)


def _evidence(support=0.0, reason=None, start=0.0, end=0.5):
    return NoteEvidence(pitch=40, start_sec=start, end_sec=end, velocity=90, harmonic_support=support,
                        spectral_contrast=0, temporal_persistence=0, onset_agreement=0, leakage=0,
                        reason=reason)


def test_score_features_order_is_exported():
    assert SCORE_FEATURES == ('harmonic_support', 'spectral_contrast', 'temporal_persistence',
                              'onset_agreement', 'leakage', 'log_duration')


@pytest.mark.parametrize('start,end', [(0.0, 0.5), (1.0, 1.0), (1.0, 0.5)])
def test_score_is_finite_for_zero_features_and_degenerate_durations(start, end):
    rule = _rule(weights=[50.0] * N, bias=-3.0)
    value = note_score(rule, _evidence(0.0, start=start, end=end))
    assert math.isfinite(value) and 0.0 <= value <= 1.0


def test_score_is_none_without_score_params():
    assert note_score(RoleRule(True, .1, .2, .1, .2), _evidence(.5)) is None


def test_score_is_sigmoid_of_standardised_linear_form():
    rule = dataclasses.replace(_rule(weights=[2.0] + [0.0] * (N - 1), bias=-1.0),
                               score_mean=(0.5,) + (0.0,) * (N - 1), score_scale=(0.25,) + (1.0,) * (N - 1))
    expected = 1 / (1 + math.exp(-(2.0 * (0.75 - 0.5) / 0.25 - 1.0)))
    assert note_score(rule, _evidence(.75)) == pytest.approx(expected)


@pytest.mark.parametrize('score,decision', [(.29, 'reject'), (.30, 'uncertain'), (.39, 'uncertain'),
                                            (.40, 'keep'), (.9, 'keep')])
def test_decision_boundaries(score, decision):
    assert score_decision(_rule(), score, None) == decision


@pytest.mark.parametrize('reason', ['out_of_range', 'invalid_duration'])
def test_out_of_range_and_invalid_duration_are_never_rejected_by_score(reason):
    assert score_decision(_rule(), 0.0, reason) == 'uncertain'


def test_other_reasons_still_use_score():
    assert score_decision(_rule(), 0.0, 'pitch_outside_cqt_range_stft_only') == 'reject'


def test_invalid_score_params_raise():
    with pytest.raises(ValueError):
        RoleRule(True, .1, .2, .1, .2, score_weights=(1.0,), score_bias=0.0, score_mean=(0.0,),
                 score_scale=(1.0,), reject_below=.1, uncertain_below=.2)
    with pytest.raises(ValueError):
        _rule(reject=.5, uncertain=.4)
    with pytest.raises(ValueError):
        dataclasses.replace(_rule(), score_scale=(0.0,) * N)
    with pytest.raises(ValueError):
        dataclasses.replace(_rule(), score_bias=None)


def test_rule_is_hashable_and_to_dict_includes_score_constants():
    rule = _rule()
    hash(rule)
    config = SpectralValidatorConfig(roles={'bass': rule})
    data = config.to_dict()['roles']['bass']
    assert data['score_weights'] == list(rule.score_weights)
    assert data['reject_below'] == .3 and data['uncertain_below'] == .4
    changed = SpectralValidatorConfig(roles={'bass': dataclasses.replace(rule, reject_below=.31)})
    assert changed.to_dict() != config.to_dict()


def test_rule_without_score_params_has_unchanged_to_dict_keys():
    data = SpectralValidatorConfig(roles={'bass': RoleRule(True, .1, .2, .1, .2)}).to_dict()['roles']['bass']
    assert set(data) == {'filter_enabled', 'reject_support', 'reject_persistence',
                         'uncertain_support', 'uncertain_persistence'}


def _files(tmp_path, notes):
    sr = 22050
    t = np.arange(sr) / sr
    sf.write(tmp_path / 'a.wav', np.sin(2 * np.pi * 440 * t).astype(np.float32), sr)
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=32, name='Bass')
    inst.notes = notes
    midi.instruments.append(inst)
    midi.write(str(tmp_path / 'a.mid'))
    return tmp_path / 'a.wav', tmp_path / 'a.mid'


def test_audit_fills_note_score_and_stays_unscored(tmp_path):
    audio, midi = _files(tmp_path, [pretty_midi.Note(90, 69, .1, .8)])
    validator = SpectralMidiValidator(SpectralValidatorConfig(roles={'bass': _rule()}))
    report = validator.validate(audio, midi, 'bass', 'audit')
    assert report.notes[0].decision == 'unscored'
    assert report.notes[0].note_score is not None and math.isfinite(report.notes[0].note_score)


def test_filter_uses_score_and_keeps_out_of_range_notes(tmp_path):
    audio, midi = _files(tmp_path, [pretty_midi.Note(90, 69, .1, .8), pretty_midi.Note(90, 60, .1, .8),
                                    pretty_midi.Note(90, 69, 5.0, 5.5)])
    # Score rises with harmonic support only: the 440 Hz note keeps, 60 rejects, out_of_range uncertain.
    rule = _rule(weights=[20.0] + [0.0] * (N - 1), bias=-4.0, reject=.3, uncertain=.4)
    validator = SpectralMidiValidator(SpectralValidatorConfig(roles={'bass': rule}))
    out = tmp_path / 'out.mid'
    report = validator.validate(audio, midi, 'bass', 'filter', out)
    decisions = {(n.pitch, round(n.start_sec, 2)): (n.decision, n.reason) for n in report.notes}
    assert decisions == {(69, .1): ('keep', None), (60, .1): ('reject', None),
                         (69, 5.0): ('uncertain', 'out_of_range')}
    assert all(n.note_score is not None for n in report.notes)
    kept = pretty_midi.PrettyMIDI(str(out)).instruments[0].notes
    assert sorted((n.pitch, round(n.start, 2)) for n in kept) == [(69, .1), (69, 5.0)]


def test_old_rule_path_leaves_note_score_empty(tmp_path):
    audio, midi = _files(tmp_path, [pretty_midi.Note(90, 69, .1, .8)])
    validator = SpectralMidiValidator(SpectralValidatorConfig(roles={'bass': RoleRule(True, .1, .2, .1, .2)}))
    report = validator.validate(audio, midi, 'bass', 'filter', tmp_path / 'out.mid')
    assert report.notes[0].note_score is None and report.notes[0].decision == 'keep'


def test_algo_version_tracks_default_role_calibration():
    calibrated = any(rule.score_weights is not None for _, rule in mv._DEFAULT_ROLES)
    assert calibrated and mv._DEFAULT_PRESENCE
    # v3 adds the calibrated stem presence rule on top of the v2 note score.
    assert mv.SPECTRAL_ALGO_VERSION == 'spectral-v3'
