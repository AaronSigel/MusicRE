import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pretty_midi
import torch

from music_re.core.backends.transcriber_base import TranscriberBackend, save_transcription
from music_re.core.midi_quantize import quantize_to_timeline, build_quantize_grid
from music_re.schemas import MusicalTimeline, TranscriptionResult


ADTOF_CLASSES = ["kick", "snare", "tom", "hihat", "cymbal"]
# GM pitch mapping for the 5 classes:
# kick -> 36 (Bass Drum 1), snare -> 38 (Acoustic Snare), tom -> 47 (Low-Mid Tom),
# hihat -> 42 (Closed Hi-Hat), cymbal -> 49 (Crash Cymbal 1)
ADTOF_GM_PITCHES = [36, 38, 47, 42, 49]
DEFAULT_THRESHOLDS = [0.22, 0.20, 0.35, 0.035, 0.20]


def _pick_peaks_with_strength(
    act: np.ndarray,
    thresholds: List[float],
    fps: int = 100,
    pre_avg: float = 0.1,
    post_avg: float = 0.01,
    pre_max: float = 0.02,
    post_max: float = 0.01,
    combine: float = 0.02,
) -> Dict[str, List[Tuple[float, float]]]:
    """Pick onsets with peak activation strength for dynamic velocity mapping."""
    out: Dict[str, List[Tuple[float, float]]] = {}
    pre_avg_n = int(round(pre_avg * fps))
    post_avg_n = int(round(post_avg * fps))
    pre_max_n = int(round(pre_max * fps))
    post_max_n = int(round(post_max * fps))
    combine_n = max(1, int(round(combine * fps)))

    for i, name in enumerate(ADTOF_CLASSES):
        x = np.asarray(act[:, i], dtype=np.float32)
        if pre_avg_n or post_avg_n:
            k = np.ones(pre_avg_n + 1 + post_avg_n, dtype=np.float32)
            k /= k.size
            padded = np.pad(x, (pre_avg_n, post_avg_n), mode="edge")
            ma = np.convolve(padded, k, mode="valid").astype(np.float32, copy=False)
            proc = np.maximum(0.0, x - ma)
        else:
            proc = x

        if pre_max_n or post_max_n:
            size = pre_max_n + 1 + post_max_n
            padded = np.pad(proc, (pre_max_n, post_max_n), mode="edge")
            win = np.stack([padded[j : j + len(proc)] for j in range(size)], axis=0).max(axis=0)
            is_peak = (proc >= win) & (proc >= thresholds[i])
        else:
            is_peak = proc >= thresholds[i]

        idx = np.where(is_peak)[0]
        peaks = []
        if idx.size:
            group = [int(idx[0])]
            for j in idx[1:]:
                if j - group[-1] <= combine_n:
                    group.append(int(j))
                else:
                    best = max(group, key=lambda z: proc[z])
                    peaks.append((best / float(fps), float(proc[best])))
                    group = [int(j)]
            best = max(group, key=lambda z: proc[z])
            peaks.append((best / float(fps), float(proc[best])))
        out[name] = peaks
    return out


def _map_velocities(per_class: Dict[str, List[Tuple[float, float]]], lo: int = 45, hi: int = 127) -> Dict[str, List[Tuple[float, int]]]:
    """Map continuous peak strengths into MIDI velocities per class."""
    vel = {}
    for name, peaks in per_class.items():
        if not peaks:
            vel[name] = []
            continue
        strengths = np.array([s for _t, s in peaks], dtype=np.float64)
        p95 = float(np.percentile(strengths, 95)) or 1.0
        vs = np.clip(np.round(lo + (hi - lo) * np.clip(strengths / p95, 0.0, 1.0)), lo, hi).astype(int)
        vel[name] = [(t, int(v)) for (t, _s), v in zip(peaks, vs)]
    return vel


class ADTOFDrumTranscriber(TranscriberBackend):
    """Deep-learning drum transcriber powered by ADTOF-pytorch (5 drum classes, dynamic velocity)."""

    def __init__(self, device: str = "auto", thresholds: Optional[List[float]] = None):
        super().__init__(backend_name="adtof", mode="drums")
        self.device = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
        self.thresholds = thresholds or list(DEFAULT_THRESHOLDS)
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from adtof_pytorch import (
            calculate_n_bins,
            create_frame_rnn_model,
            get_default_weights_path,
            load_pytorch_weights,
        )
        model = create_frame_rnn_model(calculate_n_bins())
        weights_path = get_default_weights_path()
        if not weights_path or not Path(weights_path).is_file():
            raise FileNotFoundError(f"ADTOF pretrained weights not found at: {weights_path}")
        load_pytorch_weights(model, weights_path, strict=False)
        model.to(self.device)
        model.eval()
        self._model = model

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
        stem_name: Optional[str] = None,
    ) -> TranscriptionResult:
        build_quantize_grid(grid_subdivisions=grid_subdivisions, max_duration=0)
        parameters = dict(grid_subdivisions=grid_subdivisions, quantize_ends=quantize_ends, engine="adtof")
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

        self._ensure_model()
        from adtof_pytorch import load_audio_for_model
        audio = load_audio_for_model(str(in_p)).to(self.device)

        with torch.no_grad():
            pred = self._model(audio)
        if pred.ndim == 3:
            pred = pred[0].cpu().numpy()
        else:
            pred = pred.cpu().numpy()

        peaks = _pick_peaks_with_strength(pred, self.thresholds)
        vel_events = _map_velocities(peaks)

        note_duration = 0.08
        for i, name in enumerate(ADTOF_CLASSES):
            pitch = ADTOF_GM_PITCHES[i]
            for onset, vel in vel_events.get(name, []):
                drum_inst.notes.append(
                    pretty_midi.Note(
                        velocity=vel,
                        pitch=pitch,
                        start=onset,
                        end=onset + note_duration,
                    )
                )

        drum_inst.notes.sort(key=lambda n: n.start)
        pm.instruments.append(drum_inst)

        raw_pm = pretty_midi.PrettyMIDI(initial_tempo=effective_bpm) if save_raw else None
        if raw_pm is not None:
            raw_inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
            raw_inst.notes = [pretty_midi.Note(v.velocity, v.pitch, v.start, v.end) for v in drum_inst.notes]
            raw_pm.instruments.append(raw_inst)

        if quantize:
            pm = quantize_to_timeline(
                pm=pm,
                timeline=timeline,
                bpm=effective_bpm,
                grid_subdivisions=grid_subdivisions,
                quantize_ends=quantize_ends,
                min_duration=0.04,
            )

        return save_transcription(pm, out_p, raw_pm, self.backend_name, parameters)
