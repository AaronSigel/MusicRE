from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import pytest

from music_re.core.backends.separator_base import SeparatorBackend
from music_re.core.backends import get_separator_backend, register_separator_backend

class DummySeparator(SeparatorBackend):
    def __init__(self, device: str = "auto", **kwargs):
        super().__init__(backend_name="dummy", model_name="dummy_model", device=device)
        self.loaded = False

    @property
    def supported_stems(self) -> List[str]:
        return ["vocals", "drums", "bass", "other"]

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def separate(
        self,
        input_audio: Union[Path, str],
        output_dir: Union[Path, str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Path]:
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)
        res = {}
        for s in self.supported_stems:
            p = out_d / f"{s}.wav"
            p.write_bytes(b"RIFF\x00\x00")
            res[s] = p
        return res

def test_abstract_separator_cannot_instantiate():
    with pytest.raises(TypeError):
        SeparatorBackend(backend_name="base", model_name="base_model")

def test_registry_registration_and_lookup():
    register_separator_backend("dummy", DummySeparator)
    backend = get_separator_backend("dummy", device="cpu")
    assert isinstance(backend, DummySeparator)
    assert backend.device == "cpu"
    assert backend.backend_name == "dummy"
    assert backend.supported_stems == ["vocals", "drums", "bass", "other"]

def test_registry_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown separator backend"):
        get_separator_backend("nonexistent_backend_name")
