import contextlib
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Optional, Union
import pretty_midi

from music_re.core.backends.transcriber_base import TranscriberBackend
from music_re.core.midi_quantize import quantize_to_timeline
from music_re.schemas import MusicalTimeline


class BasicPitchTranscriber(TranscriberBackend):
    """Tonal polyphonic transcriber wrapping Spotify Basic Pitch with dynamic timeline snapping."""

    def __init__(self):
        super().__init__(backend_name="basic_pitch", mode="tonal")

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
            raise FileNotFoundError(f"Audio file not found: {in_p}")

        effective_bpm = timeline.bpm if timeline and timeline.bpm > 0 else bpm
        if effective_bpm <= 0:
            effective_bpm = 120.0

        from basic_pitch.inference import predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH

        onnx_path = Path(str(ICASSP_2022_MODEL_PATH) + ".onnx")
        model_to_use = str(onnx_path) if onnx_path.exists() else ICASSP_2022_MODEL_PATH

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            with contextlib.redirect_stdout(sys.stderr):
                predict_and_save(
                    audio_path_list=[str(in_p)],
                    output_directory=str(tmp_dir_path),
                    save_midi=True,
                    sonify_midi=False,
                    save_model_outputs=False,
                    save_notes=False,
                    model_or_model_path=model_to_use,
                    midi_tempo=effective_bpm,
                )

            generated_midis = list(tmp_dir_path.glob("*.mid"))
            if not generated_midis:
                pm = pretty_midi.PrettyMIDI(initial_tempo=effective_bpm)
                pm.instruments.append(pretty_midi.Instrument(program=0))
                pm.write(str(out_p))
                if save_raw:
                    raw_out_p = out_p.with_name(f"{out_p.stem}_raw{out_p.suffix}")
                    pm.write(str(raw_out_p))
                return out_p

            raw_midi = generated_midis[0]

            if save_raw:
                raw_out_p = out_p.with_name(f"{out_p.stem}_raw{out_p.suffix}")
                shutil.copy2(str(raw_midi), str(raw_out_p))

            pm = pretty_midi.PrettyMIDI(str(raw_midi), initial_tempo=effective_bpm)

            if quantize:
                pm = quantize_to_timeline(
                    pm=pm,
                    timeline=timeline,
                    bpm=effective_bpm,
                    grid_subdivisions=4,
                )

            pm.write(str(out_p))

        return out_p
