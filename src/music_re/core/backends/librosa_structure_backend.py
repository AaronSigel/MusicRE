from pathlib import Path
from typing import Dict, List, Optional, Union
import librosa
import numpy as np

from music_re.core.audio_io import decode_audio
from music_re.core.backends.structure_base import StructureAnalyzerBackend
from music_re.schemas import MusicalTimeline, TimelineSection

SECTION_NAMES = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]


class LibrosaStructureAnalyzer(StructureAnalyzerBackend):
    """Pure Librosa baseline structure analyzer using spectral recurrence and beat tracking."""

    def __init__(self):
        super().__init__(backend_name="librosa")

    def analyze(
        self,
        audio_path: Union[Path, str],
        stems: Optional[Dict[str, Union[Path, str]]] = None,
        device: str = "auto",
    ) -> MusicalTimeline:
        in_p = Path(audio_path).resolve()
        if not in_p.is_file():
            raise FileNotFoundError(f"Audio file not found: {in_p}")

        sr = 22050
        y_mix, _ = decode_audio(in_p, target_sr=sr, normalize_db=None)
        if y_mix.ndim > 1:
            y_mix = np.mean(y_mix, axis=1)

        duration_sec = round(float(len(y_mix) / sr), 2)
        if duration_sec <= 0.0:
            return MusicalTimeline(
                duration_sec=0.0,
                bpm=120.0,
                beats=[],
                downbeats=[],
                sections=[],
            )

        # 1. Beat tracking: prefer drums stem if provided, else use mix
        y_beats = y_mix
        if stems and "drums" in stems:
            drums_p = Path(stems["drums"]).resolve()
            if drums_p.is_file():
                try:
                    y_drums, _ = decode_audio(drums_p, target_sr=sr, normalize_db=None)
                    if y_drums.ndim > 1:
                        y_drums = np.mean(y_drums, axis=1)
                    if len(y_drums) > 0 and np.max(np.abs(y_drums)) > 1e-4:
                        y_beats = y_drums
                except Exception:
                    pass

        tempo, beat_frames = librosa.beat.beat_track(y=y_beats, sr=sr)
        bpm = round(float(np.atleast_1d(tempo)[0]), 2)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        beats = [round(float(t), 3) for t in beat_times if t <= duration_sec]

        # 2. Downbeats: default 4/4 meter (every 4 beats)
        if beats:
            downbeats = [beats[i] for i in range(0, len(beats), 4)]
        else:
            downbeats = []

        # 3. Structural segmentation
        sections: List[TimelineSection] = []
        if duration_sec < 12.0:
            # Short sample: single section
            sections.append(
                TimelineSection(
                    label="intro",
                    start_sec=0.0,
                    end_sec=duration_sec,
                    confidence=0.9,
                )
            )
        else:
            # Segment via agglomerative clustering on chroma & mfcc
            try:
                hop_length = 512
                mfcc = librosa.feature.mfcc(y=y_mix, sr=sr, hop_length=hop_length, n_mfcc=13)
                chroma = librosa.feature.chroma_cqt(y=y_mix, sr=sr, hop_length=hop_length)
                features = np.vstack([mfcc, chroma])

                # Target ~1 section per 16-24 seconds
                n_segments = max(2, min(len(SECTION_NAMES), int(duration_sec // 18) + 1))
                bound_frames = librosa.segment.agglomerative(features, k=n_segments)
                bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=hop_length).tolist()

                # Clean and clamp boundaries
                cut_points = [0.0]
                for bt in bound_times:
                    t = round(float(bt), 2)
                    if 1.0 < t < duration_sec - 1.0 and t > cut_points[-1] + 2.0:
                        cut_points.append(t)
                cut_points.append(duration_sec)

                for idx in range(len(cut_points) - 1):
                    start = cut_points[idx]
                    end = cut_points[idx + 1]
                    lbl = SECTION_NAMES[min(idx, len(SECTION_NAMES) - 1)]
                    sections.append(
                        TimelineSection(
                            label=lbl,
                            start_sec=start,
                            end_sec=end,
                            confidence=0.8,
                        )
                    )
            except Exception:
                # Fallback: divide into 2-3 even sections
                mid = round(duration_sec / 2.0, 2)
                sections = [
                    TimelineSection(label="verse", start_sec=0.0, end_sec=mid, confidence=0.7),
                    TimelineSection(label="chorus", start_sec=mid, end_sec=duration_sec, confidence=0.7),
                ]

        return MusicalTimeline(
            duration_sec=duration_sec,
            bpm=bpm,
            beats=beats,
            downbeats=downbeats,
            sections=sections,
        )
