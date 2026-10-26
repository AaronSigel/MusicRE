from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from music_re.schemas import MusicalTimeline


class TranscriberBackend(ABC):
    """Abstract interface for audio-to-MIDI transcription engines."""

    def __init__(self, backend_name: str, mode: str = "tonal"):
        self.backend_name = backend_name
        self.mode = mode  # "tonal" | "drums"

    @abstractmethod
    def transcribe(
        self,
        audio_path: Union[Path, str],
        output_midi_path: Union[Path, str],
        timeline: Optional[MusicalTimeline] = None,
        bpm: float = 120.0,
        quantize: bool = True,
        save_raw: bool = True,
    ) -> Path:
        """Transcribe audio stem to MIDI file, with optional timeline synchronization and raw preservation."""
        pass
