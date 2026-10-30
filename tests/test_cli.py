from pathlib import Path
import numpy as np
import soundfile as sf
from typer.testing import CliRunner
from music_re.cli import app

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "slice" in result.stdout
    assert "analyze" in result.stdout
    assert "split" in result.stdout
    assert "to-midi" in result.stdout
    assert "inspect-midi" in result.stdout
    assert "deconstruct" in result.stdout

def test_cli_slice_command(tmp_path: Path):
    sr = 22050
    audio = np.zeros(sr * 3)
    in_wav = tmp_path / "test.wav"
    sf.write(str(in_wav), audio, sr)
    out_wav = tmp_path / "out_slice.wav"

    result = runner.invoke(app, ["slice", str(in_wav), "--start", "0.5", "--end", "1.5", "--out", str(out_wav)])
    assert result.exit_code == 0
    assert out_wav.exists()

def test_cli_inspect_midi_command(tmp_path: Path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    mid_file = tmp_path / "test.mid"
    pm.write(str(mid_file))

    result = runner.invoke(app, ["inspect-midi", str(mid_file), "--bars", "1-2", "--bpm", "120"])
    assert result.exit_code == 0
    assert "C4" in result.stdout
    assert "1" in result.stdout

def test_cli_to_midi_refuses_overwriting_input_audio(tmp_path: Path):
    sr = 22050
    in_wav = tmp_path / "drums.wav"
    sf.write(str(in_wav), np.zeros(sr), sr)
    original = in_wav.read_bytes()

    result = runner.invoke(app, ["to-midi", str(in_wav), "--mode", "drums", "--out", str(in_wav)])
    assert result.exit_code != 0
    assert in_wav.read_bytes() == original
    assert in_wav.read_bytes().startswith(b"RIFF")


def test_cli_to_midi_refuses_hardlink_of_input_audio(tmp_path: Path):
    import os
    sr = 22050
    in_wav = tmp_path / "drums.wav"
    sf.write(str(in_wav), np.zeros(sr), sr)
    original = in_wav.read_bytes()
    link = tmp_path / "hard.mid"
    os.link(in_wav, link)

    result = runner.invoke(app, ["to-midi", str(in_wav), "--mode", "drums", "--out", str(link)])
    assert result.exit_code == 2
    assert "cannot overwrite the input audio file" in result.output
    assert in_wav.read_bytes() == original


def test_cli_to_midi_refuses_default_output_equal_to_mid_input(tmp_path: Path):
    in_file = tmp_path / "stem.mid"
    in_file.write_bytes(b"RIFF-not-really-midi")

    result = runner.invoke(app, ["to-midi", str(in_file), "--mode", "drums"])
    assert result.exit_code == 2
    assert "cannot overwrite the input audio file" in result.output
    assert not isinstance(result.exception, ValueError)
    assert in_file.read_bytes() == b"RIFF-not-really-midi"


def test_cli_to_midi_refuses_raw_output_aliasing_input(tmp_path: Path):
    sr = 22050
    in_wav = tmp_path / "foo_raw.wav"
    sf.write(str(in_wav), np.zeros(sr), sr)
    original = in_wav.read_bytes()

    result = runner.invoke(app, ["to-midi", str(in_wav), "--mode", "drums", "--out", str(tmp_path / "foo.wav")])
    assert result.exit_code == 2
    assert in_wav.read_bytes() == original
