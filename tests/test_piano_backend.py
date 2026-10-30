from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from music_re.core.backends import (
    get_transcriber_backend,
    list_transcriber_backends,
)
from music_re.core.backends.piano_backend import ByteDancePianoTranscriber
from music_re.schemas import MusicalTimeline


def test_bytedance_piano_backend_registered():
    backends = list_transcriber_backends()
    assert "bytedance_piano" in backends

    instance = get_transcriber_backend(mode="tonal", name="bytedance_piano")
    assert isinstance(instance, ByteDancePianoTranscriber)
    assert instance.backend_name == "bytedance_piano"
    assert instance.mode == "tonal"
    assert instance.device in ("cpu", "cuda")


def test_bytedance_piano_device_selection():
    with patch("torch.cuda.is_available", return_value=False):
        b_auto = ByteDancePianoTranscriber(device="auto")
        assert b_auto.device == "cpu"
        b_cuda = ByteDancePianoTranscriber(device="cuda")
        assert b_cuda.device == "cpu"

    with patch("torch.cuda.is_available", return_value=True):
        b_auto = ByteDancePianoTranscriber(device="auto")
        assert b_auto.device == "cuda"
        b_cpu = ByteDancePianoTranscriber(device="cpu")
        assert b_cpu.device == "cpu"


def test_bytedance_piano_transcribe_mocked(tmp_path: Path):
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio_data = 0.3 * np.sin(2 * np.pi * 440 * t)
    audio_path = tmp_path / "piano.wav"
    sf.write(str(audio_path), audio_data, sr)

    out_midi_path = tmp_path / "piano.mid"

    def mock_transcribe_call(audio, midi_path):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, is_drum=False, name="Acoustic Grand Piano")
        note = pretty_midi.Note(velocity=80, pitch=69, start=0.0, end=0.4)
        inst.notes.append(note)
        pm.instruments.append(inst)
        pm.write(str(midi_path))
        return {"output_dict": {}, "est_note_events": [], "est_pedal_events": []}

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = mock_transcribe_call

    transcriber = ByteDancePianoTranscriber(device="cpu")
    with patch.object(transcriber, "_get_model", return_value=mock_model):
        res = transcriber.transcribe(
            audio_path=audio_path,
            output_midi_path=out_midi_path,
            bpm=120.0,
            quantize=True,
            save_raw=True,
        )

    assert res.status == "notes"
    assert res.note_count == 1
    assert res.backend == "bytedance_piano"
    assert Path(res.midi_path).is_file()
    assert res.raw_midi_path is not None
    assert Path(res.raw_midi_path).is_file()

    pm_out = pretty_midi.PrettyMIDI(res.midi_path)
    assert len(pm_out.instruments) == 1
    assert len(pm_out.instruments[0].notes) == 1


def test_bytedance_piano_transcribe_file_not_found(tmp_path: Path):
    transcriber = ByteDancePianoTranscriber(device="cpu")
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(
            audio_path=tmp_path / "nonexistent.wav",
            output_midi_path=tmp_path / "out.mid",
        )


def test_bytedance_piano_stereo_downmix(tmp_path: Path):
    sr = 16000
    stereo_data = np.stack([np.sin(np.linspace(0, 10, sr)), np.cos(np.linspace(0, 10, sr))], axis=1).astype(np.float32)
    stereo_path = tmp_path / "stereo.wav"
    sf.write(stereo_path, stereo_data, sr)

    passed_audio = []

    def mock_transcribe(audio, midi_path):
        passed_audio.append(audio)
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        pm.write(str(midi_path))

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = mock_transcribe

    transcriber = ByteDancePianoTranscriber(device="cpu")
    with patch.object(transcriber, "_get_model", return_value=mock_model):
        transcriber.transcribe(stereo_path, tmp_path / "out.mid")

    assert len(passed_audio) == 1
    assert passed_audio[0].ndim == 1
