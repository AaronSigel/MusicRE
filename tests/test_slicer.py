from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from music_re.core.slicer import slice_audio

def test_slice_audio_valid(tmp_path: Path):
    sr = 22050
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    in_file = tmp_path / "sine.wav"
    sf.write(str(in_file), audio, sr)

    out_file = tmp_path / "slice.wav"
    res = slice_audio(in_file, out_file, start_sec=1.0, end_sec=2.5)
    assert res == out_file
    assert out_file.exists()

    data, out_sr = sf.read(str(out_file))
    assert out_sr == sr
    expected_len = int(1.5 * sr)
    assert abs(len(data) - expected_len) <= 5

def test_slice_audio_end_beyond_duration(tmp_path: Path):
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    in_file = tmp_path / "short.wav"
    sf.write(str(in_file), audio, sr)

    out_file = tmp_path / "slice_end.wav"
    slice_audio(in_file, out_file, start_sec=1.0, end_sec=5.0)
    data, out_sr = sf.read(str(out_file))
    expected_len = int(1.0 * sr)
    assert abs(len(data) - expected_len) <= 5

def test_slice_audio_invalid_range(tmp_path: Path):
    in_file = tmp_path / "empty.wav"
    sf.write(str(in_file), np.zeros(22050), 22050)
    out_file = tmp_path / "err.wav"

    with pytest.raises(ValueError):
        slice_audio(in_file, out_file, start_sec=2.0, end_sec=1.0)


def test_slice_audio_refuses_identical_input_and_output(tmp_path: Path):
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    original_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_file = tmp_path / "same.wav"
    sf.write(str(wav_file), original_audio, sr)

    file_size_before = wav_file.stat().st_size
    assert file_size_before > 0

    with pytest.raises(ValueError, match="output_path cannot be identical to input_path"):
        slice_audio(wav_file, wav_file, start_sec=0.5, end_sec=1.5)

    assert wav_file.stat().st_size == file_size_before
    data, read_sr = sf.read(str(wav_file))
    assert read_sr == sr
    np.testing.assert_allclose(data, original_audio, atol=1e-4)


def test_slice_audio_refuses_symlink_output_to_input(tmp_path: Path):
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    original_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_file = tmp_path / "original.wav"
    sf.write(str(wav_file), original_audio, sr)

    symlink_file = tmp_path / "symlink.wav"
    symlink_file.symlink_to(wav_file)

    file_size_before = wav_file.stat().st_size
    assert file_size_before > 0

    with pytest.raises(ValueError, match="output_path cannot be identical to input_path"):
        slice_audio(wav_file, symlink_file, start_sec=0.5, end_sec=1.5)

    assert wav_file.stat().st_size == file_size_before
    data, read_sr = sf.read(str(wav_file))
    assert read_sr == sr
    np.testing.assert_allclose(data, original_audio, atol=1e-4)


def test_slice_audio_refuses_hardlink_output_to_input(tmp_path: Path):
    import os

    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    original_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_file = tmp_path / "original_hl.wav"
    sf.write(str(wav_file), original_audio, sr)

    hardlink_file = tmp_path / "hardlink.wav"
    os.link(wav_file, hardlink_file)

    file_size_before = wav_file.stat().st_size
    assert file_size_before > 0

    with pytest.raises(ValueError, match="output_path cannot be identical to input_path"):
        slice_audio(wav_file, hardlink_file, start_sec=0.5, end_sec=1.5)

    assert wav_file.stat().st_size == file_size_before
    data, read_sr = sf.read(str(wav_file))
    assert read_sr == sr
    np.testing.assert_allclose(data, original_audio, atol=1e-4)


def test_slice_audio_handles_fractional_seconds_in_cli(tmp_path: Path):
    from typer.testing import CliRunner
    from music_re.cli import app

    sr = 22050
    audio = np.zeros(sr * 2)
    in_wav = tmp_path / "track.wav"
    sf.write(str(in_wav), audio, sr)

    runner = CliRunner()
    res1 = runner.invoke(app, ["slice", str(in_wav), "--start", "0.0", "--end", "0.5"])
    assert res1.exit_code == 0
    res2 = runner.invoke(app, ["slice", str(in_wav), "--start", "0.2", "--end", "0.7"])
    assert res2.exit_code == 0

    out1 = tmp_path / "track_slice_0_0.5.wav"
    out2 = tmp_path / "track_slice_0.2_0.7.wav"
    assert out1.exists()
    assert out2.exists()
    assert out1 != out2
    assert not (tmp_path / "track_slice_0_0.wav").exists()

