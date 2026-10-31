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


def test_other_role_uses_stricter_reject_threshold_than_bass(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    results = {}
    for stem in ('bass', 'other'):
        midi = _midi(tmp_path / f'{stem}.mid', [_note(60)])
        report = SpectralMidiValidator().validate(
            audio, midi, stem, 'filter', tmp_path / f'{stem}-out.mid')
        results[stem] = report.notes[0].decision
    assert results['bass'] == 'reject'
    assert results['other'] in ('keep', 'uncertain')
