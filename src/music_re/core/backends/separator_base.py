from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import torch


def resolve_device(requested_device: str = "auto") -> str:
    """Resolve device string: 'auto' selects cuda if available, else cpu."""
    req = requested_device.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested ('{requested_device}') but CUDA is not available")
    return requested_device


class SeparatorBackend(ABC):
    """Abstract base class for audio source separation engines."""

    def __init__(
        self,
        backend_name: str,
        model_name: str,
        device: str = "auto",
    ):
        self.backend_name = backend_name
        self.model_name = model_name
        self.device = resolve_device(device)

    @property
    @abstractmethod
    def supported_stems(self) -> List[str]:
        """List of stems produced by this model (e.g. ['vocals', 'drums', 'bass', 'other'])."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory/device."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory/device and release memory caches."""
        pass

    @abstractmethod
    def separate(
        self,
        input_audio: Union[Path, str],
        output_dir: Union[Path, str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Path]:
        """Perform audio source separation and write stem WAV files into output_dir.

        Returns a dictionary mapping stem name to absolute file Path on disk.
        """
        pass
