from pathlib import Path
from unittest.mock import patch
import pretty_midi
import pytest

from music_re.core.backends.basic_pitch_backend import BasicPitchTranscriber


def test_basic_pitch_transcriber_defaults():
    t = BasicPitchTranscriber()
    assert t.onset_threshold == 0.5
    assert t.frame_threshold == 0.3
    assert t.minimum_note_length == 127.7
    assert t.min_velocity == 25


def test_basic_pitch_transcriber_custom_params():
    t = BasicPitchTranscriber(
        onset_threshold=0.6,
        frame_threshold=0.4,
        minimum_note_length=150.0,
        min_velocity=30,
    )
    assert t.onset_threshold == 0.6
    assert t.frame_threshold == 0.4
    assert t.minimum_note_length == 150.0
    assert t.min_velocity == 30


def test_basic_pitch_stem_threshold_calibration(tmp_path: Path):
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"dummy")
    out_midi = tmp_path / "out.mid"

    captured_kwargs = {}

    def fake_predict_and_save(**kwargs):
        captured_kwargs.update(kwargs)
        out_dir = Path(kwargs["output_directory"])
        midi_file = out_dir / "test_basic_pitch.mid"
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
        pm.instruments.append(inst)
        pm.write(str(midi_file))

    t = BasicPitchTranscriber()

    # Test "other" stem
    with patch("basic_pitch.inference.predict_and_save", side_effect=fake_predict_and_save):
        t.transcribe(audio_path, out_midi, stem_name="other", quantize=False, save_raw=False)
    assert captured_kwargs["onset_threshold"] == 0.65
    assert captured_kwargs["frame_threshold"] == 0.45
    assert captured_kwargs["minimum_note_length"] == 180.0

    # Test isolated tonal stem: vocals
    captured_kwargs.clear()
    with patch("basic_pitch.inference.predict_and_save", side_effect=fake_predict_and_save):
        t.transcribe(audio_path, out_midi, stem_name="vocals", quantize=False, save_raw=False)
    assert captured_kwargs["onset_threshold"] == 0.5
    assert captured_kwargs["frame_threshold"] == 0.3
    assert captured_kwargs["minimum_note_length"] == 127.7

    # Test isolated tonal stem: bass
    captured_kwargs.clear()
    with patch("basic_pitch.inference.predict_and_save", side_effect=fake_predict_and_save):
        t.transcribe(audio_path, out_midi, stem_name="bass", quantize=False, save_raw=False)
    assert captured_kwargs["onset_threshold"] == 0.5
    assert captured_kwargs["frame_threshold"] == 0.3
    assert captured_kwargs["minimum_note_length"] == 127.7

    # Test isolated tonal stem: guitar
    captured_kwargs.clear()
    with patch("basic_pitch.inference.predict_and_save", side_effect=fake_predict_and_save):
        t.transcribe(audio_path, out_midi, stem_name="guitar", quantize=False, save_raw=False)
    assert captured_kwargs["onset_threshold"] == 0.5
    assert captured_kwargs["frame_threshold"] == 0.3
    assert captured_kwargs["minimum_note_length"] == 127.7


def test_basic_pitch_post_filter_notes(tmp_path: Path):
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"dummy")
    out_midi = tmp_path / "out.mid"

    def fake_predict_and_save(**kwargs):
        out_dir = Path(kwargs["output_directory"])
        midi_file = out_dir / "test_basic_pitch.mid"
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        # Note 1: Valid note (vel=80, dur=0.5s) -> Keep
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
        # Note 2: Ghost note with low velocity (vel=15 < 25, dur=0.5s) -> Filter out
        inst.notes.append(pretty_midi.Note(velocity=15, pitch=62, start=0.5, end=1.0))
        # Note 3: Parasitic note with short duration (vel=90, dur=0.05s < 0.08s) -> Filter out
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=64, start=1.0, end=1.05))
        # Note 4: Boundary note (vel=25, dur=0.10s > 0.08s) -> Keep
        inst.notes.append(pretty_midi.Note(velocity=25, pitch=65, start=1.2, end=1.30))
        pm.instruments.append(inst)
        pm.write(str(midi_file))

    t = BasicPitchTranscriber(min_velocity=25)
    with patch("basic_pitch.inference.predict_and_save", side_effect=fake_predict_and_save):
        t.transcribe(audio_path, out_midi, quantize=False, save_raw=True)

    res_pm = pretty_midi.PrettyMIDI(str(out_midi))
    notes = res_pm.instruments[0].notes
    assert len(notes) == 2
    assert [n.pitch for n in notes] == [60, 65]

    # Verify raw copy also has the filtered notes
    raw_midi = out_midi.with_name("out_raw.mid")
    raw_pm = pretty_midi.PrettyMIDI(str(raw_midi))
    raw_notes = raw_pm.instruments[0].notes
    assert len(raw_notes) == 2
    assert [n.pitch for n in raw_notes] == [60, 65]
