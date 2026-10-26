from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

from music_re.schemas import MusicalTimeline


class StructureAnalyzerBackend(ABC):
    """Abstract interface for musical structure, beat, and downbeat extraction."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    @abstractmethod
    def analyze(
        self,
        audio_path: Union[Path, str],
        stems: Optional[Dict[str, Union[Path, str]]] = None,
        device: str = "auto",
    ) -> MusicalTimeline:
        """Extract musical structure timeline from audio file and optional pre-separated stems."""
        pass
