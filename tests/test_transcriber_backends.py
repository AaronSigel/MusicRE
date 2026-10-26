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
