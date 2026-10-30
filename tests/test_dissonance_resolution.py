from pathlib import Path
from unittest.mock import patch

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch

from music_re.core.backends.basic_pitch_backend import BasicPitchTranscriber
from music_re.core.backends.vocal_crepe_backend import CrepeVocalTranscriber
from music_re.core.midi_project import (
    build_combined_midi_project,
    normalize_instrument_pitch_bends,
    resolve_harmonic_dissonance,
)


def test_basic_pitch_strips_pitch_bends_for_instrumental_stems(tmp_path: Path):
    """Test that BasicPitchTranscriber strips pitch bends by default for instrumental stems."""
    audio_path = tmp_path / "inst.wav"
    audio_path.write_bytes(b"dummy")
    out_midi = tmp_path / "inst.mid"

    def fake_predict_and_save(**kwargs):
        out_dir = Path(kwargs["output_directory"])
        midi_file = out_dir / "inst_basic_pitch.mid"
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=25, name="Guitar")
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
        # Basic Pitch static +33 cent detune (pitch bend 1365)
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=1365, time=0.0))
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=1365, time=0.5))
        pm.instruments.append(inst)
        pm.write(str(midi_file))

    t = BasicPitchTranscriber()

    # Transcribing guitar stem strips static pitch bends
    with patch("basic_pitch.inference.predict_and_save", side_effect=fake_predict_and_save):
        t.transcribe(audio_path, out_midi, stem_name="guitar", quantize=False, save_raw=True)

    pm_clean = pretty_midi.PrettyMIDI(str(out_midi))
    assert len(pm_clean.instruments[0].pitch_bends) == 0

    raw_midi = out_midi.with_name("inst_raw.mid")
    pm_raw = pretty_midi.PrettyMIDI(str(raw_midi))
    assert len(pm_raw.instruments[0].pitch_bends) == 0

    # Explicitly enabled pitch bends are kept
    out_midi_kept = tmp_path / "inst_kept.mid"
    with patch("basic_pitch.inference.predict_and_save", side_effect=fake_predict_and_save):
        t.transcribe(
            audio_path,
            out_midi_kept,
            stem_name="guitar",
            quantize=False,
            save_raw=False,
            include_pitch_bends=True,
        )

    pm_kept = pretty_midi.PrettyMIDI(str(out_midi_kept))
    assert len(pm_kept.instruments[0].pitch_bends) == 2


def test_crepe_vocal_clamps_pitch_bends_to_vibrato_range(tmp_path: Path):
    """Test that CrepeVocalTranscriber clamps pitch bends to musical vibrato range (±4096)."""
    sr = 16000
    duration = 0.5
    audio_data = np.zeros(int(sr * duration), dtype=np.float32)
    audio_path = tmp_path / "vocal_extreme.wav"
    sf.write(str(audio_path), audio_data, sr)
    out_midi = tmp_path / "vocal_extreme.mid"

    def mock_bp_transcribe(audio_path, output_midi_path, *args, **kwargs):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=54, is_drum=False, name="Vocals")
        # Note at MIDI 69 (A4, 440 Hz)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=69, start=0.1, end=0.4))
        pm.instruments.append(inst)
        pm.write(str(output_midi_path))
        from music_re.schemas import TranscriptionResult
        return TranscriptionResult(
            status="notes",
            midi_path=str(output_midi_path),
            note_count=1,
            backend="basic_pitch",
        )

    n_frames = 51
    # Pitch tracker has wild octave / portamento excursion (+5 semitones -> ~587 Hz)
    mock_pitch = torch.full((1, n_frames), 587.33)
    mock_periodicity = torch.full((1, n_frames), 0.95)

    transcriber = CrepeVocalTranscriber(device="cpu", max_pitch_bend=4096)

    with patch("music_re.core.backends.basic_pitch_backend.BasicPitchTranscriber.transcribe", side_effect=mock_bp_transcribe), \
         patch("torchcrepe.predict", return_value=(mock_pitch, mock_periodicity)):
        res = transcriber.transcribe(
            audio_path=audio_path,
            output_midi_path=out_midi,
            quantize=False,
            save_raw=False,
        )

    pm = pretty_midi.PrettyMIDI(res.midi_path)
    inst = pm.instruments[0]
    assert len(inst.pitch_bends) > 0

    pb_values = [pb.pitch for pb in inst.pitch_bends]
    # All pitch bends must be clamped within ±4096
    assert all(-4096 <= val <= 4096 for val in pb_values)
    assert max(pb_values) == 4096


def test_harmonic_semitone_clash_mitigation_ghost_overtone(tmp_path: Path):
    """Test that weak ghost overtones clashing with simultaneous notes by 1 semitone are filtered."""
    piano_midi = tmp_path / "piano.mid"
    pm_piano = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst_piano = pretty_midi.Instrument(program=0, name="Piano")
    # Dominant C4 (60), strong velocity and full duration
    inst_piano.notes.append(pretty_midi.Note(velocity=95, pitch=60, start=0.0, end=1.0))
    pm_piano.instruments.append(inst_piano)
    pm_piano.write(str(piano_midi))

    guitar_midi = tmp_path / "guitar.mid"
    pm_guitar = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst_guitar = pretty_midi.Instrument(program=25, name="Guitar")
    # Clashing C#4 (61), low velocity ghost bleed note
    inst_guitar.notes.append(pretty_midi.Note(velocity=35, pitch=61, start=0.1, end=0.4))
    # Legitimate non-clashing guitar note later
    inst_guitar.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=1.5, end=2.0))
    pm_guitar.instruments.append(inst_guitar)
    pm_guitar.write(str(guitar_midi))

    out_combined = tmp_path / "combined.mid"
    build_combined_midi_project(
        stem_midis={"piano": piano_midi, "guitar": guitar_midi},
        output_path=out_combined,
        bpm=120.0,
    )

    pm_res = pretty_midi.PrettyMIDI(str(out_combined))
    piano_track = next(i for i in pm_res.instruments if i.name.lower() == "piano")
    guitar_track = next(i for i in pm_res.instruments if i.name.lower() == "guitar")

    assert len(piano_track.notes) == 1
    assert piano_track.notes[0].pitch == 60
    # Ghost overtone in guitar track filtered out; legitimate note retained
    assert len(guitar_track.notes) == 1
    assert guitar_track.notes[0].pitch == 64


def test_harmonic_semitone_clash_mitigation_nudged_to_unison(tmp_path: Path):
    """Test that non-ghost clashing notes are nudged to consonant unison to eliminate minor 2nd harshness."""
    piano_midi = tmp_path / "piano.mid"
    pm_piano = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst_piano = pretty_midi.Instrument(program=0, name="Piano")
    inst_piano.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=1.0))
    pm_piano.instruments.append(inst_piano)
    pm_piano.write(str(piano_midi))

    other_midi = tmp_path / "other.mid"
    pm_other = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst_other = pretty_midi.Instrument(program=81, name="Other")
    # Substantial note (vel=80, dur=0.9s) clashing at 61 (C#4)
    inst_other.notes.append(pretty_midi.Note(velocity=80, pitch=61, start=0.05, end=0.95))
    pm_other.instruments.append(inst_other)
    pm_other.write(str(other_midi))

    out_combined = tmp_path / "combined.mid"
    build_combined_midi_project(
        stem_midis={"piano": piano_midi, "other": other_midi},
        output_path=out_combined,
        bpm=120.0,
    )

    pm_res = pretty_midi.PrettyMIDI(str(out_combined))
    other_track = next(i for i in pm_res.instruments if i.name.lower() == "other")

    assert len(other_track.notes) == 1
    # Clashing note nudged to 60 (consonant unison with piano)
    assert other_track.notes[0].pitch == 60


def test_pitch_bend_reset_normalization_at_note_endings():
    """Test that pitch bends reset to 0 at note endings so bends do not leak into subsequent notes."""
    inst = pretty_midi.Instrument(program=54, name="Vocals")
    # Note 1: 0.0 to 1.0s
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    # Note 2: 1.5 to 2.5s (should play in tune without leaked bend)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=62, start=1.5, end=2.5))

    # Vibrato bend during Note 1
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=2048, time=0.5))

    normalize_instrument_pitch_bends(inst)

    # Check that a reset bend (pitch=0) was added at Note 1 ending
    reset_bends = [pb for pb in inst.pitch_bends if pb.pitch == 0 and abs(pb.time - 1.0) < 1e-4]
    assert len(reset_bends) == 1


def test_static_detune_stripped_in_combined_project(tmp_path: Path):
    """Test that static pitch bend detune (e.g. constant 1365 = +33 cents) is stripped in combined project."""
    bass_midi = tmp_path / "bass.mid"
    pm_bass = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst_bass = pretty_midi.Instrument(program=33, name="Bass")
    inst_bass.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=1.0))
    # Constant +33 cents detune across frames
    inst_bass.pitch_bends.append(pretty_midi.PitchBend(pitch=1365, time=0.0))
    inst_bass.pitch_bends.append(pretty_midi.PitchBend(pitch=1365, time=0.5))
    pm_bass.instruments.append(inst_bass)
    pm_bass.write(str(bass_midi))

    out_combined = tmp_path / "combined.mid"
    build_combined_midi_project(
        stem_midis={"bass": bass_midi},
        output_path=out_combined,
        bpm=120.0,
    )

    pm_res = pretty_midi.PrettyMIDI(str(out_combined))
    bass_track = next(i for i in pm_res.instruments if i.name.lower() == "bass")
    assert len(bass_track.notes) == 1
    # Pitch bends stripped due to static detune
    assert len(bass_track.pitch_bends) == 0
