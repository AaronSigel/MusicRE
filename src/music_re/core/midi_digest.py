from pathlib import Path
from typing import List, Union
import pretty_midi

from music_re.schemas import MidiDigest, NoteEvent


def generate_midi_digest(
    midi_path: Union[str, Path],
    bpm: float,
    start_bar: int = 1,
    end_bar: int = 8,
    instrument_type: str = "tonal",
) -> MidiDigest:
    """Generate compact bar-indexed digest of MIDI notes for LLM context."""
    p = Path(midi_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"MIDI file not found: {p}")

    if bpm <= 0:
        raise ValueError(f"Invalid BPM: {bpm}")

    if start_bar < 1 or end_bar < start_bar:
        raise ValueError(f"Invalid bar range: {start_bar}-{end_bar}")

    pm = pretty_midi.PrettyMIDI(str(p))

    note_events: List[NoteEvent] = []

    # Gather notes from all instruments or matching type
    for inst in pm.instruments:
        for note in inst.notes:
            total_beats = (note.start * bpm) / 60.0
            bar = int(total_beats // 4) + 1
            if bar < start_bar or bar > end_bar:
                continue

            beat_in_bar = round((total_beats % 4) + 1.0, 3)
            duration_beats = round(((note.end - note.start) * bpm) / 60.0, 3)
            note_name = pretty_midi.note_number_to_name(note.pitch)

            note_events.append(
                NoteEvent(
                    note=note_name,
                    pitch=int(note.pitch),
                    bar=bar,
                    beat_in_bar=beat_in_bar,
                    duration_beats=max(0.01, duration_beats),
                    velocity=int(note.velocity),
                )
            )

    # Sort notes by bar, beat_in_bar, pitch
    note_events.sort(key=lambda n: (n.bar, n.beat_in_bar, n.pitch))

    return MidiDigest(
        midi_path=str(p),
        bpm=round(bpm, 2),
        instrument_type=instrument_type,
        total_notes=len(note_events),
        bars_analyzed=f"{start_bar}-{end_bar}",
        notes=note_events,
    )
