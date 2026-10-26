from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import soundfile as sf
import pytest

from music_re.core.backends.separator_base import SeparatorBackend
from music_re.core.backends import register_separator_backend, list_separator_backends
from music_re.core.separator import separate_stems
from music_re.workspace import WorkspaceManager


class CustomPluginSeparator(SeparatorBackend):
    """Example third-party separation plugin (e.g., candidate BS-RoFormer / MSST)."""

    def __init__(self, device: str = "auto", **kwargs):
        super().__init__(backend_name="plugin_separator", model_name="plugin_v1", device=device)
        self.is_loaded = False

    @property
    def supported_stems(self) -> List[str]:
        return ["vocals", "instrumental"]

    def load(self) -> None:
        self.is_loaded = True

    def unload(self) -> None:
        self.is_loaded = False

    def separate(
        self,
        input_audio: Union[Path, str],
        output_dir: Union[Path, str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Path]:
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)
        results = {}
        # Produce 2 stems
        sr = 22050
        y = np.zeros(sr, dtype=np.float32)
        for s in self.supported_stems:
            p = out_d / f"{s}.wav"
            sf.write(str(p), y, sr)
            results[s] = p
        return results


def test_custom_plugin_registration_and_execution(tmp_path: Path):
    register_separator_backend("plugin_separator", CustomPluginSeparator)
    assert "plugin_separator" in list_separator_backends()

    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_wav = tmp_path / "input.wav"
    sf.write(str(track_wav), np.zeros(22050, dtype=np.float32), 22050)

    res = separate_stems(
        track_wav,
        workspace=ws,
        backend_name="plugin_separator",
    )
    assert res.model_used == "plugin_v1"
    assert "vocals" in res.stems
    assert "instrumental" in res.stems
    assert Path(res.stems["vocals"]).exists()
    assert Path(res.stems["instrumental"]).exists()


def test_custom_plugin_memory_unload():
    plugin = CustomPluginSeparator(device="cpu")
    plugin.load()
    assert plugin.is_loaded is True
    plugin.unload()
    assert plugin.is_loaded is False
