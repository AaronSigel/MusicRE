from pathlib import Path
from typing import Dict, List, Optional, Union

from music_re.core.backends.separator_base import resolve_device
from music_re.core.backends.structure_base import StructureAnalyzerBackend
from music_re.schemas import MusicalTimeline, TimelineSection


class AllInOneStructureAnalyzer(StructureAnalyzerBackend):
    """Adapter for All-In-One-Infer neural structure analysis with Demucs stems reuse."""

    def __init__(self):
        super().__init__(backend_name="allin1")

    def analyze(
        self,
        audio_path: Union[Path, str],
        stems: Optional[Dict[str, Union[Path, str]]] = None,
        device: str = "auto",
    ) -> MusicalTimeline:
        in_p = Path(audio_path).resolve()
        if not in_p.is_file():
            raise FileNotFoundError(f"Audio file not found: {in_p}")

        try:
            import allin1
        except ImportError as e:
            raise ImportError(
                "all-in-one-infer is not installed. Install with `pip install all-in-one-infer` "
                "or use the 'librosa' structure backend."
            ) from e

        resolved_device = resolve_device(device)

        # Reuse pre-separated Demucs stems directory if provided
        demucs_dir = None
        if stems and "vocals" in stems:
            stem_sample = Path(stems["vocals"]).resolve()
            if stem_sample.is_file():
                # Directory containing the stems (e.g. <track_dir>/stems/htdemucs)
                demucs_dir = stem_sample.parent

        # Call All-In-One-Infer API
        kwargs = {"device": resolved_device}
        if demucs_dir and demucs_dir.is_dir():
            kwargs["demucs_dir"] = str(demucs_dir)

        result = allin1.analyze(paths=[str(in_p)], **kwargs)
        if isinstance(result, list) and len(result) > 0:
            res = result[0]
        else:
            res = result

        beats = [round(float(b), 3) for b in getattr(res, "beats", [])]
        downbeats = [round(float(d), 3) for d in getattr(res, "downbeats", [])]
        bpm = round(float(getattr(res, "bpm", 120.0)), 2)

        raw_segments = getattr(res, "segments", [])
        sections: List[TimelineSection] = []
        for seg in raw_segments:
            sections.append(
                TimelineSection(
                    label=str(getattr(seg, "label", "segment")).lower(),
                    start_sec=round(float(getattr(seg, "start", 0.0)), 3),
                    end_sec=round(float(getattr(seg, "end", 0.0)), 3),
                    confidence=1.0,
                )
            )

        duration = beats[-1] if beats else 0.0
        if sections and sections[-1].end_sec > duration:
            duration = sections[-1].end_sec

        return MusicalTimeline(
            duration_sec=duration,
            bpm=bpm,
            beats=beats,
            downbeats=downbeats,
            sections=sections,
        )
