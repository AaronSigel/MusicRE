from pathlib import Path
from unittest.mock import patch
import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from music_re.core.backends.basic_pitch_backend import BasicPitchTranscriber
from music_re.core.dereverb import suppress_reverb_and_tails


def test_dereverb_impulse_with_decaying_tail():
    """Verify sharp impulse peak is preserved while decaying reverb tail is attenuated."""
    sr = 22050
    duration_s = 1.0
    n_samples = int(sr * duration_s)
    y = np.zeros(n_samples, dtype=np.float32)

    # Sharp initial attack impulse at sample 500
    impulse_idx = 500
    expected_peak = 0.95
    y[impulse_idx] = expected_peak

    # Decaying reverberant tail starting after the attack
    tail_start = 600
    tail_len = int(0.5 * sr)
    t_tail = np.arange(tail_len) / sr
    decay_tail = (
        0.03 * np.exp(-3.0 * t_tail) * np.sin(2 * np.pi * 440 * t_tail)
        + 0.005 * np.exp(-1.0 * t_tail) * np.random.RandomState(42).randn(tail_len)
    ).astype(np.float32)
    y[tail_start : tail_start + tail_len] = decay_tail

    processed = suppress_reverb_and_tails(
        y,
        sr=sr,
        decay_threshold_db=-24.0,
        release_ms=80.0,
    )

    # Verify initial attack peak preservation
    attack_peak_orig = float(np.max(np.abs(y[:tail_start])))
    attack_peak_proc = float(np.max(np.abs(processed[:tail_start])))
    assert np.isclose(attack_peak_proc, attack_peak_orig, atol=1e-3)
    assert np.isclose(attack_peak_proc, expected_peak, atol=1e-3)

    # Verify tail energy attenuation
    tail_eval_start = tail_start + int(0.1 * sr)
    tail_eval_end = tail_start + tail_len
    rms_tail_orig = float(np.sqrt(np.mean(y[tail_eval_start:tail_eval_end] ** 2)))
    rms_tail_proc = float(np.sqrt(np.mean(processed[tail_eval_start:tail_eval_end] ** 2)))

    # Expect significant tail attenuation (at least 6 dB reduction)
    attenuation_db = 20 * np.log10(rms_tail_proc / rms_tail_orig)
    assert attenuation_db < -6.0
    assert rms_tail_proc < rms_tail_orig


def test_dereverb_shape_preservation_and_finiteness():
    """Verify mono and stereo (n, 2) shapes are preserved and all outputs are finite."""
    sr = 22050
    n = 2205

    # 1. Mono array shape (n,)
    mono = np.random.RandomState(0).randn(n).astype(np.float32) * 0.1
    mono[100] = 0.8
    out_mono = suppress_reverb_and_tails(mono, sr=sr)
    assert out_mono.shape == (n,)
    assert out_mono.dtype == np.float32
    assert np.all(np.isfinite(out_mono))

    # 2. Stereo array shape (n, 2)
    stereo = np.column_stack([mono, mono * 0.5])
    out_stereo = suppress_reverb_and_tails(stereo, sr=sr)
    assert out_stereo.shape == (n, 2)
    assert out_stereo.dtype == np.float32
    assert np.all(np.isfinite(out_stereo))

    # 3. Stereo transposed shape (2, n)
    stereo_t = np.row_stack([mono, mono * 0.5])
    out_stereo_t = suppress_reverb_and_tails(stereo_t, sr=sr)
    assert out_stereo_t.shape == (2, n)
    assert np.all(np.isfinite(out_stereo_t))


def test_dereverb_edge_cases():
    """Verify handling of empty array, all zeros, and non-finite values."""
    # Empty array
    empty = np.array([], dtype=np.float32)
    out_empty = suppress_reverb_and_tails(empty)
    assert out_empty.size == 0

    # All-zero silence
    zeros = np.zeros(1000, dtype=np.float32)
    out_zeros = suppress_reverb_and_tails(zeros)
    assert out_zeros.shape == zeros.shape
    assert np.all(out_zeros == 0.0)

    # Very short array
    short = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out_short = suppress_reverb_and_tails(short)
    assert out_short.shape == short.shape
    assert np.all(np.isfinite(out_short))

    # NaN / Inf raises ValueError
    nan_arr = np.array([0.1, np.nan, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or Inf"):
        suppress_reverb_and_tails(nan_arr)


def test_basic_pitch_dereverb_integration(tmp_path: Path):
    """Verify Basic Pitch applies dereverberation on 'other' stem or when requested."""
    sr = 22050
    audio_wav = tmp_path / "test_synth.wav"
    t = np.arange(sr) / sr
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(audio_wav), sig, sr)

    out_midi = tmp_path / "out.mid"

    captured_paths = []
    captured_files_existed = []

    def fake_predict_and_save(**kwargs):
        captured_paths.extend(kwargs["audio_path_list"])
        captured_files_existed.append(Path(kwargs["audio_path_list"][0]).is_file())
        out_dir = Path(kwargs["output_directory"])
        midi_file = out_dir / "test_synth_basic_pitch.mid"
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
        pm.instruments.append(inst)
        pm.write(str(midi_file))

    t_transcriber = BasicPitchTranscriber()

    # Case 1: stem_name == "other" triggers dereverb
    with patch("basic_pitch.inference.predict_and_save", side_effect=fake_predict_and_save):
        t_transcriber.transcribe(
            audio_wav,
            out_midi,
            stem_name="other",
            quantize=False,
            save_raw=False,
        )
    assert len(captured_paths) == 1
    # Cleaned/dereverbed WAV was passed instead of original audio path
    assert "dereverbed" in captured_paths[0]
    assert captured_files_existed[0] is True

    # Case 2: suppress_reverb=True explicitly triggers dereverb for any stem
    captured_paths.clear()
    with patch("basic_pitch.inference.predict_and_save", side_effect=fake_predict_and_save):
        t_transcriber.transcribe(
            audio_wav,
            out_midi,
            stem_name="vocals",
            suppress_reverb=True,
            quantize=False,
            save_raw=False,
        )
    assert len(captured_paths) == 1
    assert "dereverbed" in captured_paths[0]

    # Case 3: normal stem without dereverb passes original audio file
    captured_paths.clear()
    with patch("basic_pitch.inference.predict_and_save", side_effect=fake_predict_and_save):
        t_transcriber.transcribe(
            audio_wav,
            out_midi,
            stem_name="vocals",
            suppress_reverb=False,
            quantize=False,
            save_raw=False,
        )
    assert len(captured_paths) == 1
    assert captured_paths[0] == str(audio_wav.resolve())


def test_dereverb_nonstandard_sample_count():
    # Test unusual length to verify normalized frame interpolation
    sr = 22050
    audio = np.random.randn(12345).astype(np.float32) * 0.1
    cleaned = suppress_reverb_and_tails(audio, sr=sr)
    assert cleaned.shape == audio.shape
    assert np.all(np.isfinite(cleaned))
