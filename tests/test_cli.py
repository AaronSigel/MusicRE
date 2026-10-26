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
