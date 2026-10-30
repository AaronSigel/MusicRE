import functools
import inspect
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from music_re.schemas import MusicalTimeline, TranscriptionResult


class TranscriberBackend(ABC):
    """Abstract interface for audio-to-MIDI transcription engines."""

    def __init__(self, backend_name: str, mode: str = "tonal"):
        self.backend_name = backend_name
        self.mode = mode  # "tonal" | "drums"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        impl = cls.__dict__.get("transcribe")
        if impl is None or getattr(impl, "__isabstractmethod__", False):
            return

        signature = inspect.signature(impl)

        @functools.wraps(impl)
        def transcribe(self, audio_path, output_midi_path, *args, **kwargs):
            bound = signature.bind(self, audio_path, output_midi_path, *args, **kwargs)
            bound.apply_defaults()
            ensure_distinct_output(audio_path, output_midi_path,
                                   save_raw=bool(bound.arguments.get("save_raw", False)))
            return impl(self, audio_path, output_midi_path, *args, **kwargs)

        cls.transcribe = transcribe

    @abstractmethod
    def transcribe(
        self,
        audio_path: Union[Path, str],
        output_midi_path: Union[Path, str],
        timeline: Optional[MusicalTimeline] = None,
        bpm: float = 120.0,
        quantize: bool = True,
        save_raw: bool = True,
        grid_subdivisions: int = 4,
        quantize_ends: bool = True,
        stem_name: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio stem to MIDI file, with optional timeline synchronization and raw preservation."""
        pass


def raw_output_path(output_midi_path) -> Path:
    """Path of the raw (unquantized) side output written next to the main MIDI."""
    out = Path(output_midi_path).resolve()
    return out.with_name(f'{out.stem}_raw{out.suffix}')


def _same_file(a, b) -> bool:
    a, b = Path(a), Path(b)
    if a.resolve() == b.resolve():
        return True
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def ensure_distinct_output(audio_path, output_midi_path, save_raw: bool = False) -> None:
    """Refuse output paths (main or raw) that alias the input audio file or each other."""
    if _same_file(audio_path, output_midi_path):
        raise ValueError("output_midi_path cannot be identical to audio_path")
    if save_raw:
        raw = raw_output_path(output_midi_path)
        if _same_file(audio_path, raw):
            raise ValueError("raw output path cannot be identical to audio_path")
        if _same_file(output_midi_path, raw):
            raise ValueError("raw output path cannot be identical to output_midi_path")


def save_transcription(pm, output_path, raw_pm, backend, parameters) -> TranscriptionResult:
    """Write validated MIDI atomically; an empty successful inference is explicit."""
    import tempfile
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    count = sum(len(i.notes) for i in pm.instruments)
    raw_out = raw_output_path(out) if raw_pm is not None else None
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        staged = Path(temp) / 'result.mid'
        pm.write(str(staged))
        if raw_pm is not None:
            raw_staged = Path(temp) / 'raw.mid'
            raw_pm.write(str(raw_staged))
            raw_staged.replace(raw_out)
        staged.replace(out)
    return TranscriptionResult(status='notes' if count else 'no_notes', midi_path=str(out),
                               raw_midi_path=str(raw_out) if raw_out else None,
                               note_count=count, backend=backend, parameters=parameters)
