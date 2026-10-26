from pathlib import Path

import pretty_midi
import pytest

from music_re.core.backends.basic_pitch_backend import BasicPitchTranscriber


@pytest.mark.parametrize('failure', ['missing', 'corrupt', 'exception'])
def test_basic_pitch_failure_preserves_previous_output(tmp_path, monkeypatch, failure):
    import basic_pitch.inference
    audio = tmp_path / 'song.wav'
    audio.write_bytes(b'input')
    out = tmp_path / 'song.mid'
    out.write_bytes(b'previous valid output')
    def predict_and_save(**kwargs):
        if failure == 'exception':
            raise RuntimeError('inference failed')
        if failure == 'corrupt':
            (Path(kwargs['output_directory']) / 'song.mid').write_bytes(b'invalid')
    monkeypatch.setattr(basic_pitch.inference, 'predict_and_save', predict_and_save)
    with pytest.raises(Exception):
        BasicPitchTranscriber().transcribe(audio, out)
    assert out.read_bytes() == b'previous valid output'
    assert not (tmp_path / 'song_raw.mid').exists()


def test_valid_empty_midi_has_explicit_no_notes_state(tmp_path, monkeypatch):
    import basic_pitch.inference
    audio = tmp_path / 'song.wav'
    audio.write_bytes(b'input')
    def predict_and_save(**kwargs):
        pretty_midi.PrettyMIDI().write(str(Path(kwargs['output_directory']) / 'empty.mid'))
    monkeypatch.setattr(basic_pitch.inference, 'predict_and_save', predict_and_save)
    result = BasicPitchTranscriber().transcribe(audio, tmp_path / 'empty.mid', grid_subdivisions=8)
    assert result.status == 'no_notes'
    assert result.note_count == 0
    assert Path(result.midi_path).is_file()
    assert Path(result.raw_midi_path).is_file()
