import contextlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Union
import pretty_midi

from music_re.core.midi_quantize import quantize_to_timeline
from music_re.schemas import MusicalTimeline


def clean_and_quantize_midi(
    input_midi_path: Union[str, Path],
    output_midi_path: Union[str, Path],
    bpm: float = 120.0,
    min_duration: float = 0.05,
    quantize_grid: float = 0.25,  # 0.25 = 16th note in 4/4
    timeline: Optional[MusicalTimeline] = None,
) -> Path:
    """Filter out noise artifacts, deduplicate overlapping notes, and quantize onsets (statically or via timeline)."""
    in_p = Path(input_midi_path).resolve()
    out_p = Path(output_midi_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    pm = pretty_midi.PrettyMIDI(str(in_p), initial_tempo=bpm)
    quantized_pm = quantize_to_timeline(
        pm=pm,
        timeline=timeline,
        bpm=bpm,
        grid_subdivisions=4,
        min_duration=min_duration,
    )
    quantized_pm.write(str(out_p))
    return out_p


def transcribe_tonal(
    audio_path: Union[str, Path],
    output_midi_path: Union[str, Path],
    bpm: float = 120.0,
    save_raw_copy: bool = True,
    timeline: Optional[MusicalTimeline] = None,
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

        clean_and_quantize_midi(raw_midi, out_p, bpm=bpm, timeline=timeline)

    return out_p
