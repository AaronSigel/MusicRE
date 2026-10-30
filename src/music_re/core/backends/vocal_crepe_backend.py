import copy
from pathlib import Path
import tempfile
from typing import Optional, Union

import numpy as np
import pretty_midi
import torch
import torchcrepe

from music_re.core.audio_io import decode_audio
from music_re.core.backends.basic_pitch_backend import BasicPitchTranscriber
from music_re.core.backends.transcriber_base import TranscriberBackend, save_transcription
from music_re.core.midi_quantize import build_quantize_grid, quantize_to_timeline
from music_re.schemas import MusicalTimeline, TranscriptionResult


class CrepeVocalTranscriber(TranscriberBackend):
    """Monophonic vocal transcriber combining Basic Pitch notes with TorchCREPE continuous pitch bends."""

    def __init__(
        self,
        device: str = "auto",
        hop_length_ms: float = 10.0,
        periodicity_threshold: float = 0.25,
    ):
        super().__init__(backend_name="crepe_vocal", mode="tonal")
        self.device = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
        self.hop_length_ms = hop_length_ms
        self.periodicity_threshold = periodicity_threshold

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

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_midi = Path(tmp_dir) / "basic_pitch_out.mid"
            bp = BasicPitchTranscriber()
            bp.transcribe(
                audio_path=in_p,
                output_midi_path=tmp_midi,
                timeline=timeline,
                bpm=effective_bpm,
                quantize=False,
                save_raw=False,
            )
            pm = pretty_midi.PrettyMIDI(str(tmp_midi), initial_tempo=effective_bpm)

        if not pm.instruments:
            # Ensure instrument exists so pitch bends can attach
            pm.instruments.append(pretty_midi.Instrument(program=54, is_drum=False, name="Vocals"))

        audio, sr = decode_audio(in_p, target_sr=16000, normalize_db=None)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        hop_length = int(round(sr * (self.hop_length_ms / 1000.0)))
        if hop_length <= 0:
            hop_length = 160

        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            sr,
            hop_length=hop_length,
            fmin=50.0,
            fmax=1000.0,
            model="tiny",
            return_periodicity=True,
            device=self.device,
        )

        periodicity = torchcrepe.filter.median(periodicity, 3)
        pitch = torchcrepe.filter.median(pitch, 3)

        pitch_np = pitch.squeeze(0).detach().cpu().numpy()
        periodicity_np = periodicity.squeeze(0).detach().cpu().numpy()
        times = np.arange(len(pitch_np)) * (hop_length / sr)

        for inst in pm.instruments:
            if inst.is_drum:
                continue
            inst.pitch_bends = []
            for note in inst.notes:
                mask = (
                    (times >= note.start)
                    & (times <= note.end)
                    & (periodicity_np > self.periodicity_threshold)
                )
                idx_matches = np.where(mask)[0]
                for idx in idx_matches:
                    f0 = float(pitch_np[idx])
                    if not np.isfinite(f0) or f0 <= 0:
                        continue
                    f0_note = pretty_midi.hz_to_note_number(f0)
                    semitone_diff = f0_note - note.pitch
                    pb_val = int(np.clip(np.round(semitone_diff * 4096.0 / 2.0), -8192, 8191))
                    t = float(times[idx])
                    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=pb_val, time=t))

            inst.pitch_bends.sort(key=lambda pb: pb.time)

        raw_pm = copy.deepcopy(pm) if save_raw else None

        if quantize:
            pm = quantize_to_timeline(
                pm=pm,
                timeline=timeline,
                bpm=effective_bpm,
                grid_subdivisions=grid_subdivisions,
                quantize_ends=quantize_ends,
            )

        return save_transcription(
            pm,
            out_p,
            raw_pm,
            self.backend_name,
            dict(
                grid_subdivisions=grid_subdivisions,
                quantize_ends=quantize_ends,
                device=self.device,
                hop_length_ms=self.hop_length_ms,
                periodicity_threshold=self.periodicity_threshold,
                model="crepe_tiny",
            ),
        )
