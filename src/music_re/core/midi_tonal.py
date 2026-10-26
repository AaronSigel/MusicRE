from pathlib import Path
from typing import Optional, Union

import pretty_midi

from music_re.core.midi_quantize import quantize_to_timeline
from music_re.schemas import MusicalTimeline


def clean_and_quantize_midi(input_midi_path: Union[str, Path], output_midi_path: Union[str, Path],
                            bpm: float = 120.0, min_duration: float = 0.05,
                            quantize_grid: float = 0.25,
                            timeline: Optional[MusicalTimeline] = None,
                            quantize_ends: bool = True) -> Path:
    """Clean a MIDI file; quantize_grid is expressed in fractions of a beat."""
    if quantize_grid <= 0 or not (1 / quantize_grid).is_integer():
        raise ValueError('quantize_grid must be the reciprocal of a positive integer')
    out = Path(output_midi_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(str(input_midi_path), initial_tempo=bpm)
    quantize_to_timeline(pm, timeline=timeline, bpm=bpm, grid_subdivisions=int(1 / quantize_grid),
                         min_duration=min_duration, quantize_ends=quantize_ends).write(str(out))
    return out


def transcribe_tonal(audio_path: Union[str, Path], output_midi_path: Union[str, Path],
                     bpm: float = 120.0, save_raw_copy: bool = True,
                     timeline: Optional[MusicalTimeline] = None,
                     grid_subdivisions: int = 4, quantize_ends: bool = True) -> Path:
    """Compatibility wrapper around the canonical tonal adapter."""
    from music_re.core.backends.basic_pitch_backend import BasicPitchTranscriber
    result = BasicPitchTranscriber().transcribe(audio_path, output_midi_path, timeline=timeline,
                                               bpm=bpm, save_raw=save_raw_copy,
                                               grid_subdivisions=grid_subdivisions,
                                               quantize_ends=quantize_ends)
    return Path(result.midi_path)
