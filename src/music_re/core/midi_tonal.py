import contextlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, List, Union
import pretty_midi


def clean_and_quantize_midi(
    input_midi_path: Union[str, Path],
    output_midi_path: Union[str, Path],
    bpm: float = 120.0,
    min_duration: float = 0.05,
    quantize_grid: float = 0.25,  # 0.25 = 16th note in 4/4
) -> Path:
    """Filter out noise artifacts, deduplicate/truncate overlapping same-pitch notes, and quantize onsets."""
    in_p = Path(input_midi_path).resolve()
    out_p = Path(output_midi_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    pm = pretty_midi.PrettyMIDI(str(in_p), initial_tempo=bpm)
    grid_sec = (60.0 / bpm) * quantize_grid

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue

        # 1. Filter out micro-noise and quantize onsets
        valid_notes: List[pretty_midi.Note] = []
        for note in instrument.notes:
            duration = note.end - note.start
            if duration < min_duration:
                continue

            snapped_start = round(note.start / grid_sec) * grid_sec
            snapped_end = max(snapped_start + min_duration, note.end)

            valid_notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=snapped_start,
                    end=snapped_end,
                )
            )

        # 2. Group notes by pitch to eliminate polyphonic same-pitch overlaps/conflicts
        pitch_groups: Dict[int, List[pretty_midi.Note]] = {}
        for n in valid_notes:
            pitch_groups.setdefault(n.pitch, []).append(n)

        cleaned_notes: List[pretty_midi.Note] = []
        for pitch, p_notes in pitch_groups.items():
            # Sort by start time, then end time
            p_notes.sort(key=lambda x: (x.start, x.end))
            resolved: List[pretty_midi.Note] = []

            for n in p_notes:
                if not resolved:
                    resolved.append(n)
                    continue

                prev = resolved[-1]
                if n.start < prev.end:
                    # Duplicate onset or micro-gap -> merge
                    if abs(n.start - prev.start) < min_duration:
                        prev.end = max(prev.end, n.end)
                        prev.velocity = max(prev.velocity, n.velocity)
                    else:
                        # Truncate previous note before next onset
                        prev.end = n.start
                        if prev.end > prev.start:
                            resolved.append(n)
                        else:
                            # Previous note was completely consumed, replace it
                            resolved[-1] = n
                else:
                    resolved.append(n)

            cleaned_notes.extend(resolved)

        # 3. Sort all notes chronologically
        cleaned_notes.sort(key=lambda n: (n.start, n.pitch))
        instrument.notes = cleaned_notes

    pm.write(str(out_p))
    return out_p


def transcribe_tonal(
    audio_path: Union[str, Path],
    output_midi_path: Union[str, Path],
    bpm: float = 120.0,
    save_raw_copy: bool = True,
) -> Path:
    """Transcribe melodic audio stem to MIDI using Basic Pitch, save raw unquantized copy, and clean result."""
    in_p = Path(audio_path).resolve()
    out_p = Path(output_midi_path).resolve()
    if not in_p.is_file():
        raise FileNotFoundError(f"Audio file not found: {in_p}")

    out_p.parent.mkdir(parents=True, exist_ok=True)

    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH

    onnx_path = Path(str(ICASSP_2022_MODEL_PATH) + ".onnx")
    model_to_use = str(onnx_path) if onnx_path.exists() else ICASSP_2022_MODEL_PATH

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
                model_or_model_path=model_to_use,
                midi_tempo=bpm,
            )

        generated_midis = list(tmp_dir_path.glob("*.mid"))
        if not generated_midis:
            # Fallback: create empty MIDI
            pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            pm.instruments.append(pretty_midi.Instrument(program=0))
            pm.write(str(out_p))
            if save_raw_copy:
                raw_out_p = out_p.with_name(f"{out_p.stem}_raw{out_p.suffix}")
                pm.write(str(raw_out_p))
            return out_p

        raw_midi = generated_midis[0]

        if save_raw_copy:
            raw_out_p = out_p.with_name(f"{out_p.stem}_raw{out_p.suffix}")
            shutil.copy2(str(raw_midi), str(raw_out_p))

        clean_and_quantize_midi(raw_midi, out_p, bpm=bpm)

    return out_p
