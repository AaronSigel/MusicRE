from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch

from music_re.core.backends import get_transcriber_backend, list_transcriber_backends
from music_re.core.backends.vocal_crepe_backend import CrepeVocalTranscriber


def test_crepe_vocal_backend_registered():
    backends = list_transcriber_backends()
    assert "crepe_vocal" in backends

    instance = get_transcriber_backend(mode="tonal", name="crepe_vocal")
    assert isinstance(instance, CrepeVocalTranscriber)
    assert instance.backend_name == "crepe_vocal"
    assert instance.mode == "tonal"
    assert instance.device in ("cpu", "cuda")


def test_crepe_vocal_device_selection():
    with patch("torch.cuda.is_available", return_value=False):
        b_auto = CrepeVocalTranscriber(device="auto")
        assert b_auto.device == "cpu"
        b_cuda = CrepeVocalTranscriber(device="cuda")
        assert b_cuda.device == "cpu"

    with patch("torch.cuda.is_available", return_value=True):
        b_auto = CrepeVocalTranscriber(device="auto")
        assert b_auto.device == "cuda"
        b_cpu = CrepeVocalTranscriber(device="cpu")
        assert b_cpu.device == "cpu"


def test_crepe_vocal_transcribe_synthetic_vibrato(tmp_path: Path):
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz (MIDI 69) with 5 Hz vibrato (~15 Hz depth)
    f_inst = 440.0 + 15.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    audio_data = (0.5 * np.sin(phase)).astype(np.float32)

    audio_path = tmp_path / "vocal_vibrato.wav"
    sf.write(str(audio_path), audio_data, sr)

    out_midi_path = tmp_path / "vocal_vibrato.mid"

    transcriber = CrepeVocalTranscriber(device="cpu")
    res = transcriber.transcribe(
        audio_path=audio_path,
        output_midi_path=out_midi_path,
        bpm=120.0,
        quantize=True,
        save_raw=True,
    )

    assert res.status == "notes"
    assert res.note_count >= 1
    assert res.backend == "crepe_vocal"
    assert Path(res.midi_path).is_file()
    assert res.raw_midi_path is not None
    assert Path(res.raw_midi_path).is_file()

    pm_out = pretty_midi.PrettyMIDI(res.midi_path)
    assert len(pm_out.instruments) >= 1
    inst = pm_out.instruments[0]
    assert len(inst.notes) >= 1
    assert len(inst.pitch_bends) > 0

    pb_values = [pb.pitch for pb in inst.pitch_bends]
    assert min(pb_values) >= -8192
    assert max(pb_values) <= 8191
    # Vibrato creates both positive and negative pitch bend swings
    assert any(v > 0 for v in pb_values)
    assert any(v < 0 for v in pb_values)


def test_crepe_vocal_pitch_bend_math_mocked(tmp_path: Path):
    sr = 16000
    duration = 0.5
    audio_data = np.zeros(int(sr * duration), dtype=np.float32)
    audio_path = tmp_path / "dummy_vocal.wav"
    sf.write(str(audio_path), audio_data, sr)
    out_midi_path = tmp_path / "dummy_vocal.mid"

    def mock_bp_transcribe(audio_path, output_midi_path, *args, **kwargs):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=54, is_drum=False, name="Vocals")
        # Note at MIDI 69 (A4, 440Hz)
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

    # Mock crepe predict: return pitch at 440 Hz (0 bend) and 466.16 Hz (+1 semitone = +2048 bend)
    n_frames = 51
    mock_pitch = torch.full((1, n_frames), 440.0)
    # Frames around 0.3s (idx 29..31) -> +1 semitone -> ~466.16 Hz (survives median filter)
    mock_pitch[0, 29:32] = 466.1637615
    mock_periodicity = torch.full((1, n_frames), 0.9)

    transcriber = CrepeVocalTranscriber(device="cpu", hop_length_ms=10.0, periodicity_threshold=0.25)

    with patch("music_re.core.backends.basic_pitch_backend.BasicPitchTranscriber.transcribe", side_effect=mock_bp_transcribe), \
         patch("torchcrepe.predict", return_value=(mock_pitch, mock_periodicity)):
        res = transcriber.transcribe(
            audio_path=audio_path,
            output_midi_path=out_midi_path,
            quantize=False,
            save_raw=False,
        )

    pm = pretty_midi.PrettyMIDI(res.midi_path)
    inst = pm.instruments[0]
    assert len(inst.pitch_bends) > 0

    pb_at_30 = next(pb for pb in inst.pitch_bends if abs(pb.time - 0.30) < 1e-4)
    # 1 semitone diff * 4096 / 2 = 2048
    assert abs(pb_at_30.pitch - 2048) <= 2


def test_crepe_vocal_file_not_found(tmp_path: Path):
    transcriber = CrepeVocalTranscriber(device="cpu")
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(
            audio_path=tmp_path / "nonexistent.wav",
            output_midi_path=tmp_path / "out.mid",
        )
