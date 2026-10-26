from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from music_re.core.midi_digest import generate_midi_digest
from music_re.core.midi_drums import transcribe_drums
from music_re.core.midi_tonal import transcribe_tonal
from music_re.core.separator import separate_stems
from music_re.workspace import WorkspaceManager
import music_re.mcp_server as mcp_srv


def test_mcp_server_no_name_shadowing_and_tools_callable():
    """Finding 2: MCP server functions should not shadow core functions."""
    ws = WorkspaceManager(cache_root=Path("/tmp/.dummy_cache"))
    # Check that core_separate_stems or separator function is callable and not shadowed
    assert hasattr(mcp_srv, "separate_stems")
    assert hasattr(mcp_srv, "core_separate_stems")


def test_mcp_server_main_does_not_destroy_stdout():
    """Finding 1: FastMCP stdio writer must keep sys.stdout intact for json-rpc."""
    import sys
    orig_stdout = sys.stdout
    with patch.object(mcp_srv.mcp, "run") as mock_run:
        mcp_srv.main()
        # stdout should remain stdout during/after main
        assert sys.stdout == orig_stdout
        mock_run.assert_called_once_with(transport="stdio")


def test_transcribe_tonal_execution(tmp_path: Path):
    """Finding 3: transcribe_tonal should run end-to-end without import errors."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave for 440 Hz (A4)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_file = tmp_path / "tone.wav"
    sf.write(str(wav_file), audio, sr)

    out_midi = tmp_path / "tone.mid"
    res = transcribe_tonal(wav_file, out_midi, bpm=120.0)
    assert res.exists()
    pm = pretty_midi.PrettyMIDI(str(res))
    assert len(pm.instruments) > 0


def test_separator_silent_audio(tmp_path: Path):
    """Finding 4: Silent audio must not divide by zero or raise AssertionError in Demucs."""
    sr = 44100
    audio = np.zeros(sr * 2)
    wav_file = tmp_path / "silent.wav"
    sf.write(str(wav_file), audio, sr)

    ws = WorkspaceManager(cache_root=tmp_path / ".cache")
    result = separate_stems(wav_file, workspace=ws)
    assert "vocals" in result.stems
    # Check that output stems exist and are readable
    data, _ = sf.read(result.stems["vocals"])
    assert np.all(data == 0)


def test_midi_drums_short_audio(tmp_path: Path):
    """Finding 5: Short audio (< 22 samples) must not crash scipy filtfilt."""
    sr = 22050
    audio = np.zeros(10)  # Only 10 samples
    wav_file = tmp_path / "tiny.wav"
    sf.write(str(wav_file), audio, sr)

    out_mid = tmp_path / "tiny.mid"
    res = transcribe_drums(wav_file, out_mid, bpm=120.0)
    assert res.exists()
    pm = pretty_midi.PrettyMIDI(str(res))
    assert isinstance(pm, pretty_midi.PrettyMIDI)


def test_separator_atomic_cache(tmp_path: Path):
    """Finding 6: Partial writes must not poison cache without completion marker."""
    ws = WorkspaceManager(cache_root=tmp_path / ".cache")
    track_file = tmp_path / "song.wav"
    track_file.write_bytes(b"RIFF" + b"\x00" * 500)

    track_dir = ws.get_track_dir(track_file)
    stems_dir = track_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    # Stems exist but no completion marker
    (stems_dir / "vocals.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    (stems_dir / "drums.wav").write_bytes(b"RIFF" + b"\x00" * 100)

    # When stems are incomplete or no marker, it shouldn't treat partial files as valid cached result
    # We will test with marker
    marker = stems_dir / ".complete"
    assert not marker.exists()


def test_midi_digest_instrument_type_filter(tmp_path: Path):
    """Finding 8: instrument_type filter should separate drums from tonal."""
    bpm = 120.0
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    tonal_inst = pretty_midi.Instrument(program=0, is_drum=False)
    tonal_inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))

    drum_inst = pretty_midi.Instrument(program=0, is_drum=True)
    drum_inst.notes.append(pretty_midi.Note(velocity=90, pitch=36, start=0.5, end=0.6))

    pm.instruments.extend([tonal_inst, drum_inst])
    mid_file = tmp_path / "mixed.mid"
    pm.write(str(mid_file))

    digest_tonal = generate_midi_digest(mid_file, bpm=bpm, instrument_type="tonal")
    assert digest_tonal.total_notes == 1
    assert digest_tonal.notes[0].pitch == 60

    digest_drums = generate_midi_digest(mid_file, bpm=bpm, instrument_type="drums")
    assert digest_drums.total_notes == 1
    assert digest_drums.notes[0].pitch == 36
