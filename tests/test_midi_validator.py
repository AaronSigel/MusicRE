import hashlib
import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from music_re.core.midi_validation import (
    RoleRule, SpectralMidiValidator, SpectralValidatorConfig,
)


def _audio(path, frequencies=(440.0,), seconds=1.0):
    sr = 22050
    t = np.arange(int(sr * seconds)) / sr
    y = sum(np.sin(2 * np.pi * f * t) for f in frequencies).astype(np.float32)
    sf.write(path, y, sr)
    return path


def _midi(path, notes=None):
    midi = pretty_midi.PrettyMIDI(initial_tempo=123)
    inst = pretty_midi.Instrument(program=32, name='Bass')
    inst.notes = notes or []
    midi.instruments.append(inst)
    midi.write(str(path))
    return path


def _note(pitch, start=0.1, end=0.8, velocity=87):
    return pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_is_byte_identical_unscored_and_writes_no_output(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / 'a.mid', [_note(69)])
    before = _digest(midi)
    output = tmp_path / 'must-not-exist.mid'
    report = SpectralMidiValidator().validate(audio, midi, 'bass', 'audit', output)
    assert _digest(midi) == before
    assert [n.decision for n in report.notes] == ['unscored']
    assert not output.exists()
    assert report.output_midi_path is None


def test_bass_filter_rejects_wrong_pitch_keeps_correct_and_preserves_original(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / 'a.mid', [_note(69), _note(60, .1, .8, 53)])
    before = _digest(midi)
    out = tmp_path / 'filtered.mid'
    report = SpectralMidiValidator().validate(audio, midi, 'bass', 'filter', out)
    assert {n.pitch: n.decision for n in report.notes} == {69: 'keep', 60: 'reject'}
    assert _digest(midi) == before
    notes = pretty_midi.PrettyMIDI(str(out)).instruments[0].notes
    assert [(n.pitch, n.velocity) for n in notes] == [(69, 87)]


def test_kept_note_fields_tempo_program_and_instrument_name_are_preserved(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / 'a.mid', [_note(69)])
    source_midi = pretty_midi.PrettyMIDI(str(midi))
    source = source_midi.instruments[0].notes[0]
    out = tmp_path / 'filtered.mid'
    SpectralMidiValidator().validate(audio, midi, 'bass', 'filter', out)
    result = pretty_midi.PrettyMIDI(str(out))
    kept = result.instruments[0].notes[0]
    assert (kept.pitch, kept.start, kept.end, kept.velocity) == (
        source.pitch, source.start, source.end, source.velocity)
    assert result.get_tempo_changes() == source_midi.get_tempo_changes()
    assert result.instruments[0].program == source_midi.instruments[0].program
    assert result.instruments[0].name == source_midi.instruments[0].name


def test_empty_midi_writes_valid_empty_output_and_zero_counts(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / 'empty.mid')
    out = tmp_path / 'empty-out.mid'
    report = SpectralMidiValidator().validate(audio, midi, 'bass', 'filter', out)
    assert report.counts == {}
    assert all(not instrument.notes for instrument in pretty_midi.PrettyMIDI(str(out)).instruments)


def test_polyphony_keeps_both_simultaneous_supported_notes(tmp_path):
    audio = _audio(tmp_path / 'a.wav', (440.0, 523.25))
    midi = _midi(tmp_path / 'a.mid', [_note(69), _note(72)])
    report = SpectralMidiValidator().validate(audio, midi, 'bass', 'filter', tmp_path / 'out.mid')
    assert {note.pitch: note.decision for note in report.notes} == {69: 'keep', 72: 'keep'}


def test_filter_does_not_introduce_duplicate_notes(tmp_path):
    audio = _audio(tmp_path / 'a.wav', (440.0, 523.25))
    midi = _midi(tmp_path / 'a.mid', [_note(69), _note(72)])
    out = tmp_path / 'out.mid'
    SpectralMidiValidator().validate(audio, midi, 'bass', 'filter', out)
    notes = pretty_midi.PrettyMIDI(str(out)).instruments[0].notes
    keys = {(n.pitch, n.start, n.end, n.velocity) for n in notes}
    assert len(notes) == len(keys) == 2


@pytest.mark.parametrize(('stem', 'status', 'decision'), [
    ('drums', 'skipped', 'skipped'),
    ('synth', 'unsupported', 'unsupported'),
])
def test_drums_and_unknown_stems_copy_through_with_expected_status(
    tmp_path, stem, status, decision
):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / f'{stem}.mid', [_note(60)])
    before = _digest(midi)
    out = tmp_path / f'{stem}-out.mid'
    validator = SpectralMidiValidator()
    if stem == 'synth':
        with pytest.warns(UserWarning, match='Unsupported stem'):
            report = validator.validate(audio, midi, stem, 'filter', out)
        assert report.warnings
    else:
        report = validator.validate(audio, midi, stem, 'filter', out)
    assert report.status == status
    assert report.notes[0].decision == decision
    assert _digest(out) == before


@pytest.mark.parametrize('stem', ['piano', 'vocals'])
def test_audit_only_roles_never_reject_wrong_pitch_in_filter_mode(tmp_path, stem):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / f'{stem}.mid', [_note(60)])
    report = SpectralMidiValidator().validate(audio, midi, stem, 'filter', tmp_path / 'out.mid')
    assert report.notes[0].decision == 'unscored'
    assert len(pretty_midi.PrettyMIDI(str(tmp_path / 'out.mid')).instruments[0].notes) == 1


def test_piano_filter_evidence_matches_audit_and_is_nonzero(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / 'piano.mid', [_note(69)])
    validator = SpectralMidiValidator()
    audit = validator.validate(audio, midi, 'piano', 'audit')
    filtered = validator.validate(audio, midi, 'piano', 'filter', tmp_path / 'copy.mid')
    assert filtered.notes[0].harmonic_support == audit.notes[0].harmonic_support > 0
    assert filtered.notes[0].temporal_persistence == audit.notes[0].temporal_persistence > 0


def test_rich_midi_filter_preserves_non_note_events_instruments_and_duplicates(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=100)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 0))
    midi.key_signature_changes.append(pretty_midi.KeySignature(2, 0))
    midi.lyrics.append(pretty_midi.Lyric('preserve me', 0.2))
    first = pretty_midi.Instrument(program=32, name='Bass')
    first.notes = [_note(69), _note(69), _note(60)]
    first.control_changes.append(pretty_midi.ControlChange(7, 99, 0.15))
    first.pitch_bends.append(pretty_midi.PitchBend(128, 0.25))
    second = pretty_midi.Instrument(program=40, name='Second')
    second.notes = [_note(69)]
    midi.instruments.extend([first, second])
    source_path = tmp_path / 'rich.mid'
    midi.write(str(source_path))
    source = pretty_midi.PrettyMIDI(str(source_path))
    out = tmp_path / 'rich-out.mid'
    SpectralMidiValidator().validate(audio, source_path, 'bass', 'filter', out)
    result = pretty_midi.PrettyMIDI(str(out))
    assert result.resolution == source.resolution == 480
    assert result.get_tempo_changes() == source.get_tempo_changes()
    assert [(x.numerator, x.denominator, x.time) for x in result.time_signature_changes] == [
        (x.numerator, x.denominator, x.time) for x in source.time_signature_changes]
    assert [(x.key_number, x.time) for x in result.key_signature_changes] == [
        (x.key_number, x.time) for x in source.key_signature_changes]
    assert [(x.text, x.time) for x in result.lyrics] == [(x.text, x.time) for x in source.lyrics]
    assert len(result.instruments) == len(source.instruments) == 2
    for actual, expected in zip(result.instruments, source.instruments):
        assert (actual.program, actual.name, actual.is_drum) == (
            expected.program, expected.name, expected.is_drum)
        assert [(x.number, x.value, x.time) for x in actual.control_changes] == [
            (x.number, x.value, x.time) for x in expected.control_changes]
        assert [(x.pitch, x.time) for x in actual.pitch_bends] == [
            (x.pitch, x.time) for x in expected.pitch_bends]
    assert [n.pitch for n in result.instruments[0].notes] == [69, 69]
    assert [n.pitch for n in result.instruments[1].notes] == [69]


@pytest.mark.parametrize('pitch,frequency', [(33, 55.0), (28, 41.203)])
def test_quiet_a1_and_harmonic_low_e1_are_kept(tmp_path, pitch, frequency):
    if pitch == 33:
        audio = _audio(tmp_path / 'a.wav', (frequency,))
        # Match the requested quiet fundamental amplitude.
        sr = 22050
        t = np.arange(sr) / sr
        sf.write(audio, (0.003 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sr)
    else:
        sr = 22050
        t = np.arange(sr) / sr
        y = sum((np.sin(2 * np.pi * frequency * harmonic * t) / harmonic
                 for harmonic in range(1, 6))).astype(np.float32)
        audio = tmp_path / 'a.wav'
        sf.write(audio, y, sr)
    midi = _midi(tmp_path / 'low.mid', [_note(pitch)])
    report = SpectralMidiValidator().validate(audio, midi, 'bass', 'filter', tmp_path / 'out.mid')
    assert report.notes[0].decision == 'keep'


@pytest.mark.parametrize('mode,out_path', [
    ('filter', None),
    ('filter', 'same'),
    ('off', None),
])
def test_invalid_filter_arguments_raise_value_error(tmp_path, mode, out_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / 'a.mid')
    out = midi if out_path == 'same' else None
    with pytest.raises(ValueError):
        SpectralMidiValidator().validate(audio, midi, 'bass', mode, out)


def test_config_dict_is_deterministic_json_serializable_and_threshold_sensitive():
    config = SpectralValidatorConfig()
    encoded = json.dumps(config.to_dict(), sort_keys=True)
    assert encoded == json.dumps(config.to_dict(), sort_keys=True)
    changed = SpectralValidatorConfig(roles=(
        ('bass', RoleRule(True, .11, .20, .10, .20)),
    ))
    assert config.to_dict() != changed.to_dict()


def test_threshold_rule_other_uses_stricter_reject_threshold_than_bass(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    config = SpectralValidatorConfig(roles={'bass': RoleRule(True, .10, .20, .10, .20),
                                            'other': RoleRule(True, .06, .10, .10, .20)})
    results = {}
    for stem in ('bass', 'other'):
        midi = _midi(tmp_path / f'{stem}.mid', [_note(60)])
        report = SpectralMidiValidator(config).validate(
            audio, midi, stem, 'filter', tmp_path / f'{stem}-out.mid')
        results[stem] = report.notes[0].decision
    assert results['bass'] == 'reject'
    assert results['other'] in ('keep', 'uncertain')


def test_default_other_role_uses_calibrated_score(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / 'other.mid', [_note(69), _note(60)])
    report = SpectralMidiValidator().validate(audio, midi, 'other', 'filter', tmp_path / 'out.mid')
    assert {n.pitch: n.decision for n in report.notes} == {69: 'keep', 60: 'reject'}
    assert all(n.note_score is not None for n in report.notes)
