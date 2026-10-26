import contextlib
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import soundfile as sf

from music_re.core.audio_io import decode_audio
from music_re.core.backends.separator_base import resolve_device
from music_re.core.backends.structure_base import StructureAnalyzerBackend
from music_re.schemas import MusicalTimeline, TimelineSection


class AllInOneStructureAnalyzer(StructureAnalyzerBackend):
    """All-In-One-Infer adapter; all upstream intermediates are disposable."""

    def __init__(self, model_name: str = "harmonix-all"):
        super().__init__(backend_name="allin1")
        self.model_name = model_name

    def analyze(self, audio_path: Union[Path, str],
                stems: Optional[Dict[str, Union[Path, str]]] = None,
                device: str = "auto") -> MusicalTimeline:
        in_p = Path(audio_path).resolve()
        if not in_p.is_file():
            raise FileNotFoundError(f"Audio file not found: {in_p}")
        import allin1_infer

        audio, sr = decode_audio(in_p, target_sr=44100, normalize_db=None)
        duration = len(audio) / sr
        with tempfile.TemporaryDirectory(prefix="music-re-structure-") as temp:
            root = Path(temp)
            kwargs = dict(device=resolve_device(device), model=self.model_name,
                          demix_dir=root / 'demix', spec_dir=root / 'spec',
                          keep_byproducts=True, multiprocess=False)
            if stems:
                required = {'bass', 'drums', 'other', 'vocals'}
                if not required.issubset(stems):
                    raise ValueError(f"Missing structure stems: {sorted(required - stems.keys())}")
                paths = {name: Path(stems[name]).resolve() for name in required}
                # Six-source Demucs puts guitar/piano outside 'other'. Restore
                # the four-source representation expected by the neural model.
                extra = [stems[n] for n in ('guitar', 'piano') if n in stems]
                if extra:
                    other, rate = sf.read(paths['other'], dtype='float32', always_2d=True)
                    for path in extra:
                        signal, extra_rate = sf.read(path, dtype='float32', always_2d=True)
                        if rate != extra_rate or signal.shape != other.shape:
                            raise ValueError("Incompatible six-source stem shapes or sample rates")
                        other += signal
                    paths['other'] = root / 'other.wav'
                    sf.write(paths['other'], other, rate, subtype='FLOAT')
                kwargs['stems_input'] = [allin1_infer.StemsInput(**paths, identifier='track')]
            else:
                # Use our FFmpeg decoder for consistent MP3 timing.
                decoded = root / 'input.wav'
                sf.write(decoded, audio, sr, subtype='FLOAT')
                kwargs['paths'] = [str(decoded)]
            with contextlib.redirect_stdout(sys.stderr):
                result = allin1_infer.analyze(**kwargs)
            if isinstance(result, list):
                if len(result) != 1:
                    raise RuntimeError("All-In-One-Infer returned no result or multiple results")
                result = result[0]
            if result is None:
                raise RuntimeError("All-In-One-Infer returned no result")

        sections = [TimelineSection(label=str(seg.label).lower(),
                                    start_sec=max(0.0, float(seg.start)),
                                    end_sec=min(duration, float(seg.end)),
                                    confidence=getattr(seg, 'confidence', None))
                    for seg in result.segments if float(seg.start) < duration and float(seg.end) > 0]
        # An undetected tempo is "unknown", not absent: 0.0 is the same
        # sentinel pipeline.py already falls back from (bpm > 0 check).
        return MusicalTimeline(duration_sec=duration, bpm=float(result.bpm) if result.bpm is not None else 0.0,
                               beats=[float(t) for t in result.beats if 0 <= t <= duration],
                               downbeats=[float(t) for t in result.downbeats if 0 <= t <= duration],
                               sections=sections, backend=self.backend_name, model=self.model_name)
