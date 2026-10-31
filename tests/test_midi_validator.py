import hashlib
import json

import numpy as np
import pretty_midi
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


def test_audit_is_byte_identical_and_unscored(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / 'a.mid', [_note(69)])
    before = _digest(midi)
    report = SpectralMidiValidator().validate(audio, midi, 'bass', 'audit')
    assert _digest(midi) == before
    assert [n.decision for n in report.notes] == ['unscored']
    assert report.output_midi_path is None


def test_filter_rejects_wrong_pitch_and_preserves_original_and_note_fields(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / 'a.mid', [_note(69), _note(60, .1, .8, 53)])
    original_notes = {n.pitch: n for n in pretty_midi.PrettyMIDI(str(midi)).instruments[0].notes}
    before = _digest(midi)
    out = tmp_path / 'filtered.mid'
    report = SpectralMidiValidator().validate(audio, midi, 'bass', 'filter', out)
    assert {n.pitch: n.decision for n in report.notes} == {69: 'keep', 60: 'reject'}
    assert _digest(midi) == before
    result = pretty_midi.PrettyMIDI(str(out))
    assert len(result.instruments[0].notes) == 1
    kept = result.instruments[0].notes[0]
    source = original_notes[69]
    assert (kept.pitch, kept.start, kept.end, kept.velocity) == (source.pitch, source.start, source.end, source.velocity)
    assert result.instruments[0].program == 32
    assert result.instruments[0].name == 'Bass'
    assert result.get_tempo_changes()[1][0] == pretty_midi.PrettyMIDI(str(midi)).get_tempo_changes()[1][0]


def test_empty_polyphony_and_no_duplicates(tmp_path):
    audio = _audio(tmp_path / 'a.wav', (440.0, 523.25))
    midi = _midi(tmp_path / 'a.mid', [_note(69), _note(72)])
    out = tmp_path / 'filtered.mid'
    report = SpectralMidiValidator().validate(audio, midi, 'bass', 'filter', out)
    assert [n.decision for n in report.notes] == ['keep', 'keep']
    notes = pretty_midi.PrettyMIDI(str(out)).instruments[0].notes
    assert len(notes) == len({(n.pitch, n.start, n.end, n.velocity) for n in notes}) == 2
    empty = _midi(tmp_path / 'empty.mid')
    empty_out = tmp_path / 'empty-out.mid'
    empty_report = SpectralMidiValidator().validate(audio, empty, 'bass', 'filter', empty_out)
    assert empty_report.counts == {}
    assert all(not instrument.notes for instrument in pretty_midi.PrettyMIDI(str(empty_out)).instruments)


def test_special_roles_passthrough_and_config_serialization(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    for stem in ('piano', 'vocals', 'drums', 'synth'):
        midi = _midi(tmp_path / f'{stem}.mid', [_note(60)])
        out = tmp_path / f'{stem}-out.mid'
        report = SpectralMidiValidator().validate(audio, midi, stem, 'filter', out)
        assert len(pretty_midi.PrettyMIDI(str(out)).instruments[0].notes) == 1
        if stem == 'drums':
            assert report.status == 'skipped' and report.notes[0].decision == 'skipped'
        elif stem == 'synth':
            assert report.status == 'unsupported' and report.notes[0].decision == 'unsupported'
            assert report.warnings
        else:
            assert report.notes[0].decision == 'unscored'
    config = SpectralValidatorConfig()
    assert json.dumps(config.to_dict(), sort_keys=True) == json.dumps(config.to_dict(), sort_keys=True)
    changed = SpectralValidatorConfig(roles=(('bass', RoleRule(True, .2, .2, .5, .5)),))
    assert changed.to_dict() != config.to_dict()


def test_filter_requires_distinct_output_and_supported_mode(tmp_path):
    audio = _audio(tmp_path / 'a.wav')
    midi = _midi(tmp_path / 'a.mid')
    validator = SpectralMidiValidator()
    import pytest
    with pytest.raises(ValueError):
        validator.validate(audio, midi, 'bass', 'filter')
    with pytest.raises(ValueError):
        validator.validate(audio, midi, 'bass', 'filter', midi)
    with pytest.raises(ValueError):
        validator.validate(audio, midi, 'bass', 'off')
