import contextlib
import copy
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Optional, Union
import pretty_midi

from music_re.core.backends.transcriber_base import TranscriberBackend, save_transcription
from music_re.core.midi_quantize import quantize_to_timeline, build_quantize_grid
from music_re.schemas import MusicalTimeline, TranscriptionResult


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
        grid_subdivisions: int = 4,
        quantize_ends: bool = True,
    ) -> TranscriptionResult:
        build_quantize_grid(grid_subdivisions=grid_subdivisions, max_duration=0)
        in_p = Path(audio_path).resolve()
        out_p = Path(output_midi_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not in_p.is_file():
            raise FileNotFoundError(f"Audio file not found: {in_p}")

        effective_bpm = timeline.bpm if timeline and timeline.bpm > 0 else bpm
        if effective_bpm <= 0:
            effective_bpm = 120.0

        from basic_pitch.inference import predict_and_save
        from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
        model_to_use = build_icassp_2022_model_path(FilenameSuffix.onnx)

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
            if len(generated_midis) != 1:
                raise RuntimeError('Basic Pitch did not produce exactly one MIDI output')

            raw_midi = generated_midis[0]

            pm = pretty_midi.PrettyMIDI(str(raw_midi), initial_tempo=effective_bpm)
            raw_pm = copy.deepcopy(pm) if save_raw else None

            if quantize:
                pm = quantize_to_timeline(
                    pm=pm,
                    timeline=timeline,
                    bpm=effective_bpm,
                    grid_subdivisions=grid_subdivisions,
                    quantize_ends=quantize_ends,
                )

            return save_transcription(pm, out_p, raw_pm, self.backend_name,
                                      dict(grid_subdivisions=grid_subdivisions, quantize_ends=quantize_ends,
                                           model='icassp_2022/nmp.onnx'))
