import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from music_re.core.backends.structure_base import StructureAnalyzerBackend
from music_re.schemas import MusicalTimeline, TimelineSection


class BeatThisStructureAnalyzer(StructureAnalyzerBackend):
    """Deep-learning beat & downbeat tracker using CPJKU Beat This! with optional drum phase alignment."""

    def __init__(self, device: str = "auto", checkpoint_path: str = "final0"):
        super().__init__(backend_name="beat_this")
        self.device = "cuda" if device in ("auto", "cuda") and __import__("torch").cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from beat_this.inference import File2Beats
        self._model = File2Beats(checkpoint_path=self.checkpoint_path, device=self.device, dbn=False)

    def analyze(
        self,
        audio_path: Union[str, Path],
        stems: Optional[dict] = None,
        device: str = "auto",
        **kwargs,
    ) -> MusicalTimeline:
        p = Path(audio_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Audio file not found: {p}")

        self._ensure_model()
        beats, downbeats = self._model(str(p))

        beats = [round(float(b), 4) for b in beats]
        downbeats = [round(float(d), 4) for d in downbeats]

        # Calculate BPM from beat intervals
        if len(beats) >= 2:
            intervals = np.diff(beats)
            bpm = round(float(60.0 / np.median(intervals)), 2)
        else:
            bpm = 120.0

        import soundfile as sf
        info = sf.info(str(p))
        duration_sec = round(float(info.duration), 2)

        # Single continuous section or simple intro/body/outro if sections not directly emitted by beat_this
        sections: List[TimelineSection] = []
        if downbeats and len(downbeats) >= 2:
            if downbeats[0] > 0.5:
                sections.append(TimelineSection(label="intro", start_sec=0.0, end_sec=downbeats[0]))
            sections.append(TimelineSection(label="body", start_sec=downbeats[0], end_sec=downbeats[-1]))
            if duration_sec > downbeats[-1]:
                sections.append(TimelineSection(label="outro", start_sec=downbeats[-1], end_sec=duration_sec))
        else:
            sections.append(TimelineSection(label="music", start_sec=0.0, end_sec=duration_sec))

        return MusicalTimeline(
            duration_sec=duration_sec,
            bpm=bpm,
            beats=beats,
            downbeats=downbeats,
            sections=sections,
            backend="beat_this",
            model=self.checkpoint_path,
        )
