import json
import click
import pytest
import soundfile as sf
import numpy as np
from pathlib import Path
from typer.testing import CliRunner
from music_re.cli import app
from music_re.workspace import WorkspaceManager

runner = CliRunner()

def test_cli_evaluate_fails_cleanly_on_nonexistent_or_empty_cache(tmp_path):
    dummy_wav = tmp_path / "song.wav"
    sf.write(dummy_wav, np.zeros((44100, 2)), 44100)
    fake_cache = tmp_path / "nonexistent_cache"
    
    # Should fail cleanly without creating fake_cache or corrupting it
    res = runner.invoke(app, ["evaluate", str(dummy_wav), "--cache-root", str(fake_cache)])
    assert res.exit_code != 0
    assert "No manifest or completed artifacts found" in res.output or "Error" in res.output
    # Must NOT create directories inside fake_cache if manifest was not found!
    assert not fake_cache.exists()

def test_cli_to_midi_mode_choice_validation(tmp_path):
    dummy_wav = tmp_path / "song.wav"
    sf.write(dummy_wav, np.zeros((44100, 2)), 44100)
    
    # Invalid mode 'drmus' should exit with code != 0 and raise BadParameter
    res = runner.invoke(app, ["to-midi", str(dummy_wav), "--mode", "drmus"])
    assert res.exit_code != 0
    assert isinstance(res.exception, (click.BadParameter, SystemExit))

def test_cli_inspect_midi_part_selector(tmp_path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    drum_inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(drum_inst)
    midi_file = tmp_path / "test.mid"
    pm.write(str(midi_file))
    
    # Default / explicit --part all should show drum notes
    res = runner.invoke(app, ["inspect-midi", str(midi_file), "--part", "all"])
    assert res.exit_code == 0
    assert "Total notes in range: 1" in res.output
    
    # --part tonal should report 0
    res_tonal = runner.invoke(app, ["inspect-midi", str(midi_file), "--part", "tonal"])
    assert res_tonal.exit_code == 0
    assert "Total notes in range: 0" in res_tonal.output
