import copy
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pretty_midi
import torch

from music_re.core.audio_io import decode_audio
from music_re.core.backends.transcriber_base import TranscriberBackend, save_transcription
from music_re.core.midi_quantize import quantize_to_timeline
from music_re.schemas import MusicalTimeline, TranscriptionResult


class ByteDancePianoTranscriber(TranscriberBackend):
    """Specialized polyphonic piano transcriber wrapping ByteDance piano_transcription_inference."""

    def __init__(self, device: str = "auto"):
        super().__init__(backend_name="bytedance_piano", mode="tonal")
        self.device = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
        self._model = None

    def _get_model(self):
        if self._model is None:
            import piano_transcription_inference
            self._model = piano_transcription_inference.PianoTranscription(device=self.device)
        return self._model

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
        in_p = Path(audio_path).resolve()
        out_p = Path(output_midi_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not in_p.is_file():
            raise FileNotFoundError(f"Audio file not found: {in_p}")

        effective_bpm = timeline.bpm if timeline and timeline.bpm > 0 else bpm
        if effective_bpm <= 0:
            effective_bpm = 120.0

        audio, _ = decode_audio(in_p, target_sr=16000, normalize_db=None)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        model = self._get_model()
        model.transcribe(audio, str(out_p))

        if not out_p.is_file():
            # If model didn't write file (e.g. no events or mocked without writing)
            pm = pretty_midi.PrettyMIDI(initial_tempo=effective_bpm)
        else:
            pm = pretty_midi.PrettyMIDI(str(out_p))

        raw_pm = copy.deepcopy(pm) if save_raw else None

        if quantize:
            pm = quantize_to_timeline(
                pm=pm,
                timeline=timeline,
                bpm=effective_bpm,
                grid_subdivisions=grid_subdivisions,
                quantize_ends=quantize_ends,
            )

        return save_transcription(
            pm,
            out_p,
            raw_pm,
            self.backend_name,
            dict(
                grid_subdivisions=grid_subdivisions,
                quantize_ends=quantize_ends,
                device=self.device,
                model="bytedance_piano_crnn",
            ),
        )
