from pathlib import Path
from typing import List, Optional, Tuple, Union
import pretty_midi

from music_re.core.backends.transcriber_base import TranscriberBackend
from music_re.core.midi_drums import _bandpass_filter, _detect_band_onsets
from music_re.core.midi_quantize import quantize_to_timeline
from music_re.core.audio_io import decode_audio
from music_re.schemas import MusicalTimeline


class DrumTranscriber(TranscriberBackend):
    """Multi-band energy drum transcriber with timeline synchronization."""

    def __init__(self):
        super().__init__(backend_name="drum_multiband", mode="drums")

    def transcribe(
        self,
        audio_path: Union[Path, str],
        output_midi_path: Union[Path, str],
        timeline: Optional[MusicalTimeline] = None,
        bpm: float = 120.0,
        quantize: bool = True,
        save_raw: bool = True,
    ) -> Path:
        in_p = Path(audio_path).resolve()
        out_p = Path(output_midi_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not in_p.is_file():
            raise FileNotFoundError(f"Drum audio file not found: {in_p}")

        effective_bpm = timeline.bpm if timeline and timeline.bpm > 0 else bpm
        if effective_bpm <= 0:
            effective_bpm = 120.0

        pm = pretty_midi.PrettyMIDI(initial_tempo=effective_bpm)
        drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

        sr = 22050
        y, _ = decode_audio(in_p, target_sr=sr, normalize_db=None)
        if y.ndim > 1:
            import numpy as np
            y = np.mean(y, axis=1)

        import numpy as np
        if len(y) <= 21 or np.max(np.abs(y)) < 1e-5:
            pm.instruments.append(drum_inst)
            pm.write(str(out_p))
            if save_raw:
                raw_out_p = out_p.with_name(f"{out_p.stem}_raw{out_p.suffix}")
                pm.write(str(raw_out_p))
            return out_p

        # Multi-band frequency definitions:
        bands = [
            ("kick", 36, 30.0, 140.0, 0.05),
            ("snare", 38, 220.0, 900.0, 0.07),
            ("hihat", 42, 4500.0, 10000.0, 0.04),
        ]

        all_events: List[Tuple[float, int, int]] = []
        for name, pitch, low_cut, high_cut, delta in bands:
            filtered = _bandpass_filter(y, sr, low_cut, high_cut)
            events = _detect_band_onsets(filtered, sr, pitch, delta=delta)
            all_events.extend(events)

        all_events.sort(key=lambda e: (e[0], e[1]))

        # Minimum duration for GM drum note
        note_duration = 0.08
        for onset, pitch, velocity in all_events:
            drum_inst.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=onset,
                    end=onset + note_duration,
                )
            )

        pm.instruments.append(drum_inst)

        if save_raw:
            raw_out_p = out_p.with_name(f"{out_p.stem}_raw{out_p.suffix}")
            pm.write(str(raw_out_p))

        if quantize:
            pm = quantize_to_timeline(
                pm=pm,
                timeline=timeline,
                bpm=effective_bpm,
                grid_subdivisions=4,
                min_duration=0.04,
            )

        pm.write(str(out_p))
        return out_p
