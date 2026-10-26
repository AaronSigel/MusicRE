from pathlib import Path
from typing import Dict, Optional, Union
import pretty_midi

from music_re.schemas import MusicalTimeline

GM_PROGRAM_MAP = {
    "bass": 33,      # Electric Bass (finger)
    "piano": 0,      # Acoustic Grand Piano
    "guitar": 25,    # Acoustic Guitar (steel)
    "vocals": 54,    # Synth Voice
    "other": 81,     # Lead 2 (sawtooth)
    "drums": 0,      # Channel 10 Standard Kit
}


def build_combined_midi_project(
    stem_midis: Dict[str, Union[Path, str]],
    output_path: Union[Path, str],
    timeline: Optional[MusicalTimeline] = None,
    bpm: float = 120.0,
) -> Path:
    """Combine individual stem MIDI files into a single multi-track Standard MIDI File (SMF Type 1)."""
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    effective_bpm = timeline.bpm if timeline and timeline.bpm > 0 else bpm
    combined_pm = pretty_midi.PrettyMIDI(initial_tempo=effective_bpm)

    for stem_name, stem_path in stem_midis.items():
        stem_p = Path(stem_path).resolve()
        if not stem_p.is_file():
            continue

        try:
            stem_pm = pretty_midi.PrettyMIDI(str(stem_p), initial_tempo=effective_bpm)
        except Exception:
            continue

        s_key = stem_name.lower()
        is_drum = s_key in ("drums", "percussion")
        program = GM_PROGRAM_MAP.get(s_key, 0)

        # Merge all notes from this stem's instruments into one designated track
        stem_instrument = pretty_midi.Instrument(
            program=program,
            is_drum=is_drum,
            name=stem_name.capitalize(),
        )

        for src_inst in stem_pm.instruments:
            for note in src_inst.notes:
                stem_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end,
                    )
                )

        stem_instrument.notes.sort(key=lambda n: (n.start, n.pitch))
        combined_pm.instruments.append(stem_instrument)

    combined_pm.write(str(out_p))
    return out_p
