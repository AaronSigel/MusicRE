import json
from pathlib import Path

import mido
import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from music_re.core.midi_validation import (
    PRESENCE_ROLES, PresenceRule, SpectralMidiValidator, SpectralValidatorConfig,
)

SR = 22050


def _tone(path, amplitude, seconds=2.0, frequency=440.0):
    t = np.arange(int(SR * seconds)) / SR
    sf.write(path, (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32), SR)
    return path


def _midi(path, pitches=(60, 62, 64), name='Voice'):
    midi = pretty_midi.PrettyMIDI(initial_tempo=110)
    inst = pretty_midi.Instrument(program=52, name=name)
    inst.notes = [pretty_midi.Note(velocity=90, pitch=p, start=.1 + .5 * i, end=.5 + .5 * i)
                  for i, p in enumerate(pitches)]
    midi.instruments.append(inst)
    midi.write(str(path))
    return path


def _config(**rules):
    return SpectralValidatorConfig(presence=rules or {
        role: PresenceRule(min_rel_db=-30.0, min_median_support=0.3) for role in PRESENCE_ROLES})


@pytest.fixture
def quiet(tmp_path):
    mix = _tone(tmp_path / 'mix.wav', 0.5)
    stem = _tone(tmp_path / 'stem.wav', 0.002)
    return mix, stem


@pytest.mark.parametrize('role', ['vocals', 'bass', 'other', 'guitar', 'piano'])
def test_quiet_stem_with_spurious_notes_is_absent_in_filter(tmp_path, quiet, role):
    mix, stem = quiet
    midi = _midi(tmp_path / f'{role}.mid')
    out = tmp_path / f'{role}-out.mid'
    report = SpectralMidiValidator(_config()).validate(stem, midi, role, 'filter', out, mix_audio_path=mix)
    assert report.status == 'absent'
    assert report.presence['verdict'] == 'absent'
    assert report.presence['rel_db'] < -30
    assert {n.decision for n in report.notes} == {'reject'}
    assert {n.reason for n in report.notes} == {'stem_absent'}
    assert report.counts == {'reject': 3}
    # pretty_midi drops note-less instruments on read; the track itself is kept in the file.
    assert 'Voice' in [track.name for track in mido.MidiFile(str(out)).tracks]
    assert sum(len(i.notes) for i in pretty_midi.PrettyMIDI(str(out)).instruments) == 0
    assert len(pretty_midi.PrettyMIDI(str(midi)).instruments[0].notes) == 3


def test_absent_stem_is_untouched_in_audit(tmp_path, quiet):
    mix, stem = quiet
    midi = _midi(tmp_path / 'vocals.mid')
    before = midi.read_bytes()
    report = SpectralMidiValidator(_config()).validate(stem, midi, 'vocals', 'audit', mix_audio_path=mix)
    assert report.status == 'validated'
    assert report.presence['verdict'] == 'absent'
    assert {n.decision for n in report.notes} == {'unscored'}
    assert all(n.reason != 'stem_absent' for n in report.notes)
    assert report.output_midi_path is None
    assert midi.read_bytes() == before


def test_presence_features_are_reported(tmp_path, quiet):
    mix, stem = quiet
    midi = _midi(tmp_path / 'vocals.mid')
    report = SpectralMidiValidator(_config()).validate(stem, midi, 'vocals', 'audit', mix_audio_path=mix)
    p = report.presence
    assert set(p) >= {'rel_db', 'active_fraction', 'median_support', 'verdict'}
    assert p['rel_db'] == pytest.approx(20 * np.log10(0.002 / 0.5), abs=1.0)
    assert 0.0 <= p['active_fraction'] <= 1.0
    assert p['median_support'] == pytest.approx(float(np.median([n.harmonic_support for n in report.notes])))
    json.dumps(report.model_dump(), allow_nan=False)


def test_loud_real_stem_is_present_and_filtered_normally(tmp_path):
    mix = _tone(tmp_path / 'mix.wav', 0.5)
    stem = _tone(tmp_path / 'stem.wav', 0.4)
    midi = _midi(tmp_path / 'bass.mid', pitches=(69, 60))
    report = SpectralMidiValidator(_config()).validate(stem, midi, 'bass', 'filter', tmp_path / 'o.mid',
                                                       mix_audio_path=mix)
    assert report.status == 'validated'
    assert report.presence['verdict'] == 'present'
    assert report.presence['rel_db'] > -3
    assert {n.pitch: n.decision for n in report.notes} == {69: 'keep', 60: 'reject'}


def test_supported_notes_keep_quiet_stem_present(tmp_path):
    mix = _tone(tmp_path / 'mix.wav', 0.5)
    stem = _tone(tmp_path / 'stem.wav', 0.002)
    midi = _midi(tmp_path / 'vocals.mid', pitches=(69, 69, 69))
    report = SpectralMidiValidator(_config()).validate(stem, midi, 'vocals', 'filter', tmp_path / 'o.mid',
                                                       mix_audio_path=mix)
    assert report.presence['median_support'] >= 0.3
    assert report.presence['verdict'] == 'present'
    assert report.status == 'validated'


def test_no_mix_path_never_absent(tmp_path, quiet):
    _, stem = quiet
    midi = _midi(tmp_path / 'vocals.mid')
    out = tmp_path / 'o.mid'
    report = SpectralMidiValidator(_config()).validate(stem, midi, 'vocals', 'filter', out)
    assert report.status == 'validated'
    assert report.presence['verdict'] is None
    assert report.presence['rel_db'] is None
    assert out.read_bytes() == midi.read_bytes()


def test_silent_mix_gives_no_verdict(tmp_path):
    mix = _tone(tmp_path / 'mix.wav', 0.0)
    stem = _tone(tmp_path / 'stem.wav', 0.0)
    midi = _midi(tmp_path / 'vocals.mid')
    report = SpectralMidiValidator(_config()).validate(stem, midi, 'vocals', 'filter', tmp_path / 'o.mid',
                                                       mix_audio_path=mix)
    assert report.presence['verdict'] is None
    assert report.status == 'validated'


def test_drums_are_never_presence_checked(tmp_path, quiet):
    mix, stem = quiet
    midi = _midi(tmp_path / 'drums.mid', pitches=(36, 38))
    out = tmp_path / 'o.mid'
    config = _config(drums=PresenceRule(0.0, 1.0), vocals=PresenceRule(0.0, 1.0))
    report = SpectralMidiValidator(config).validate(stem, midi, 'drums', 'filter', out, mix_audio_path=mix)
    assert report.status == 'skipped'
    assert report.presence is None
    assert out.read_bytes() == midi.read_bytes()


def test_role_without_presence_rule_is_not_checked(tmp_path, quiet):
    mix, stem = quiet
    midi = _midi(tmp_path / 'bass.mid')
    report = SpectralMidiValidator(_config(vocals=PresenceRule(-30.0, .5))).validate(
        stem, midi, 'bass', 'filter', tmp_path / 'o.mid', mix_audio_path=mix)
    assert report.presence['verdict'] is None
    assert report.status == 'validated'


def test_default_config_has_presence_for_calibrated_roles_only():
    config = SpectralValidatorConfig()
    data = config.to_dict()
    assert set(PRESENCE_ROLES) == {'vocals', 'bass', 'other', 'guitar', 'piano'}
    assert set(data['presence']) == {'vocals'}
    for rule in data['presence'].values():
        assert set(rule) == {'min_rel_db', 'min_median_support'}
    changed = SpectralValidatorConfig(presence={'vocals': PresenceRule(-10.0, .1)})
    assert changed.to_dict() != data
    assert config.presence_for('drums') is None
    hash(config)


@pytest.mark.parametrize('role', ['bass', 'other', 'guitar', 'piano'])
def test_default_uncalibrated_roles_report_no_presence_verdict(tmp_path, role):
    mix = _tone(tmp_path / 'mix.wav', 0.5)
    stem = _tone(tmp_path / 'stem.wav', 0.3)
    midi = _midi(tmp_path / f'{role}.mid')
    report = SpectralMidiValidator().validate(stem, midi, role, 'audit', mix_audio_path=mix)
    assert report.presence['rel_db'] is not None
    assert report.presence['verdict'] is None


def test_presence_rule_rejects_non_finite():
    with pytest.raises(ValueError):
        PresenceRule(float('nan'), .1)
    with pytest.raises(ValueError):
        PresenceRule(-20.0, float('inf'))
