import contextlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Union
import pretty_midi


def clean_and_quantize_midi(
    input_midi_path: Union[str, Path],
    output_midi_path: Union[str, Path],
    bpm: float = 120.0,
    min_duration: float = 0.05,
    quantize_grid: float = 0.25,  # 0.25 = 16th note in 4/4
) -> Path:
    """Filter out noise artifacts, merge overlapping notes, and quantize onsets."""
    in_p = Path(input_midi_path).resolve()
    out_p = Path(output_midi_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    pm = pretty_midi.PrettyMIDI(str(in_p), initial_tempo=bpm)
    grid_sec = (60.0 / bpm) * quantize_grid

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue

        # Sort notes by start time, then pitch
        notes = sorted(instrument.notes, key=lambda n: (n.start, n.pitch))
        cleaned_notes = []

        for note in notes:
            # 1. Filter out micro-noise
            duration = note.end - note.start
            if duration < min_duration:
                continue

            # 2. Quantize start time
            snapped_start = round(note.start / grid_sec) * grid_sec
            snapped_end = max(snapped_start + min_duration, note.end)

            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=snapped_start,
                end=snapped_end,
            )

            # 3. Merge overlapping duplicate pitches
            if cleaned_notes and cleaned_notes[-1].pitch == new_note.pitch:
                prev_note = cleaned_notes[-1]
                if new_note.start <= prev_note.end:
                    # Extend previous note
                    prev_note.end = max(prev_note.end, new_note.end)
                    prev_note.velocity = max(prev_note.velocity, new_note.velocity)
                    continue

            cleaned_notes.append(new_note)

        instrument.notes = cleaned_notes

    pm.write(str(out_p))
    return out_p


def transcribe_tonal(
    audio_path: Union[str, Path],
    output_midi_path: Union[str, Path],
    bpm: float = 120.0,
) -> Path:
    """Transcribe melodic audio stem to MIDI using Basic Pitch and clean result."""
    in_p = Path(audio_path).resolve()
    out_p = Path(output_midi_path).resolve()
    if not in_p.is_file():
        raise FileNotFoundError(f"Audio file not found: {in_p}")

    out_p.parent.mkdir(parents=True, exist_ok=True)

    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Redirect stdout to stderr for Basic Pitch inference
        with contextlib.redirect_stdout(sys.stderr):
            predict_and_save(
                audio_path_list=[str(in_p)],
                output_directory=str(tmp_dir_path),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                midi_tempo=bpm,
            )

        generated_midis = list(tmp_dir_path.glob("*.mid"))
        if not generated_midis:
            # Fallback: create empty MIDI
            pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            pm.instruments.append(pretty_midi.Instrument(program=0))
            pm.write(str(out_p))
            return out_p

        raw_midi = generated_midis[0]
        clean_and_quantize_midi(raw_midi, out_p, bpm=bpm)

    return out_p
