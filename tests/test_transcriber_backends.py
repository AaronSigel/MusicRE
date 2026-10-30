from pathlib import Path
from typing import Optional, Union
import pytest

from music_re.core.backends.transcriber_base import TranscriberBackend
from music_re.core.backends import (
    get_transcriber_backend,
    register_transcriber_backend,
    list_transcriber_backends,
)
from music_re.schemas import MusicalTimeline


class MockTranscriber(TranscriberBackend):
    def __init__(self, mode: str = "tonal"):
        super().__init__(backend_name="mock_transcriber", mode=mode)

    def transcribe(
        self,
        audio_path: Union[Path, str],
        output_midi_path: Union[Path, str],
        timeline: Optional[MusicalTimeline] = None,
        bpm: float = 120.0,
        quantize: bool = True,
        save_raw: bool = True,
    ) -> Path:
        out_p = Path(output_midi_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0")
        if save_raw:
            raw_p = out_p.with_name(f"{out_p.stem}_raw{out_p.suffix}")
            raw_p.write_bytes(b"MThd_RAW")
        return out_p


def test_abstract_transcriber_cannot_instantiate():
    with pytest.raises(TypeError):
        TranscriberBackend(backend_name="base", mode="tonal")


def test_transcriber_registry_defaults():
    tonal = get_transcriber_backend(mode="tonal")
    assert tonal.mode == "tonal"
    assert tonal.backend_name == "basic_pitch"

    drums = get_transcriber_backend(mode="drums")
    assert drums.mode == "drums"
    assert drums.backend_name == "drum_multiband"


def test_custom_transcriber_registration(tmp_path: Path):
    register_transcriber_backend("mock", MockTranscriber)
    assert "mock" in list_transcriber_backends()

    inst = get_transcriber_backend(name="mock", mode="tonal")
    assert isinstance(inst, MockTranscriber)

    in_f = tmp_path / "tone.wav"
    in_f.write_bytes(b"DATA")
    out_f = tmp_path / "tone.mid"
    res = inst.transcribe(in_f, out_f, save_raw=True)
    assert res.exists()
    assert (tmp_path / "tone_raw.mid").exists()


def _write_test_wav(path: Path) -> bytes:
    import numpy as np
    import soundfile as sf

    sr = 22050
    t = np.arange(sr) / sr
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * 220 * t), sr)
    return path.read_bytes()


def test_transcribe_refuses_overwriting_input_audio_file(tmp_path: Path):
    from music_re.core.backends.drum_backend import DrumTranscriber

    wav = tmp_path / "drums.wav"
    original = _write_test_wav(wav)

    with pytest.raises(ValueError, match="output_midi_path cannot be identical to audio_path"):
        DrumTranscriber().transcribe(wav, wav)
    assert wav.read_bytes() == original
    assert wav.read_bytes().startswith(b"RIFF")

    link = tmp_path / "link.mid"
    link.symlink_to(wav)
    with pytest.raises(ValueError, match="output_midi_path cannot be identical to audio_path"):
        DrumTranscriber().transcribe(wav, link)
    assert wav.read_bytes() == original
    assert link.is_symlink()


def test_transcribe_refuses_raw_output_aliasing_input_audio(tmp_path: Path):
    from music_re.core.backends.drum_backend import DrumTranscriber

    wav = tmp_path / "foo_raw.wav"
    original = _write_test_wav(wav)

    with pytest.raises(ValueError, match="raw output path cannot be identical to audio_path"):
        DrumTranscriber().transcribe(wav, tmp_path / "foo.wav")
    with pytest.raises(ValueError, match="raw output path cannot be identical to audio_path"):
        DrumTranscriber().transcribe(wav, tmp_path / "foo.wav", save_raw=True)
    assert wav.read_bytes() == original
    assert not (tmp_path / "foo.wav").exists()

    result = DrumTranscriber().transcribe(wav, tmp_path / "foo.wav", save_raw=False)
    assert result.raw_midi_path is None
    assert wav.read_bytes() == original


def test_transcribe_refuses_raw_output_aliasing_main_output(tmp_path: Path):
    from music_re.core.backends.drum_backend import DrumTranscriber

    wav = tmp_path / "drums.wav"
    _write_test_wav(wav)
    out = tmp_path / "out.mid"
    out.write_bytes(b"MThd")
    (tmp_path / "out_raw.mid").symlink_to(out)

    with pytest.raises(ValueError, match="raw output path cannot be identical to output_midi_path"):
        DrumTranscriber().transcribe(wav, out)


def test_transcribe_refuses_hardlink_of_input_audio(tmp_path: Path):
    import os
    from music_re.core.backends.drum_backend import DrumTranscriber

    wav = tmp_path / "drums.wav"
    original = _write_test_wav(wav)
    link = tmp_path / "hard.mid"
    os.link(wav, link)

    with pytest.raises(ValueError, match="output_midi_path cannot be identical to audio_path"):
        DrumTranscriber().transcribe(wav, link, save_raw=False)
    assert wav.read_bytes() == original
