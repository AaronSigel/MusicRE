from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pretty_midi
import soundfile as sf

from music_re.core.audio_io import decode_audio
from music_re.schemas import QualityReport
from music_re.workspace import WorkspaceManager


def compute_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Compute Signal-to-Distortion Ratio (SDR) in decibels."""
    min_len = min(len(reference), len(estimated))
    ref = reference[:min_len]
    est = estimated[:min_len]

    noise = ref - est
    ref_energy = np.sum(ref**2)
    noise_energy = np.sum(noise**2)

    if noise_energy <= 1e-12:
        return 50.0  # Cap perfect match at +50 dB
    if ref_energy <= 1e-12:
        return -50.0

    sdr = 10.0 * np.log10(ref_energy / (noise_energy + 1e-12))
    return round(float(sdr), 2)


def evaluate_quality(
    workspace: WorkspaceManager,
    file_path: Union[str, Path],
    ground_truth_stems: Optional[Union[str, Path]] = None,
    ground_truth_midi: Optional[Union[str, Path]] = None,
) -> QualityReport:
    """Evaluate audio separation and MIDI transcription quality with two-tier objective/diagnostic metrics."""
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}")

    sr = 22050
    y_mix, _ = decode_audio(p, target_sr=sr, normalize_db=None)
    if y_mix.ndim > 1:
        y_mix = np.mean(y_mix, axis=1)

    # 1. Audio Diagnostics
    peak = float(np.max(np.abs(y_mix))) if len(y_mix) > 0 else 0.0
    peak_dbfs = round(20.0 * np.log10(max(peak, 1e-9)), 2)
    clipping_samples = int(np.sum(np.abs(y_mix) >= 0.999))

    # Signal-to-noise floor estimate
    if len(y_mix) > 0:
        rms_val = np.sqrt(np.mean(y_mix**2) + 1e-12)
        noise_floor = np.percentile(np.abs(y_mix), 10) + 1e-9
        snr_estimate = round(float(20.0 * np.log10(rms_val / noise_floor)), 2)
    else:
        snr_estimate = 0.0

    audio_diagnostics = {
        "duration_sec": round(float(len(y_mix) / sr), 2),
        "peak_dbfs": peak_dbfs,
        "clipping_samples": clipping_samples,
        "snr_estimate_db": snr_estimate,
    }

    manifest = workspace.get_manifest(p)
    is_objective = False
    separation_metrics: Dict[str, Any] = {}
    transcription_metrics: Dict[str, Any] = {}

    # 2. Separation Quality (Objective vs Diagnostic)
    if ground_truth_stems and Path(ground_truth_stems).is_dir():
        gt_dir = Path(ground_truth_stems).resolve()
        is_objective = True
        sdr_by_stem: Dict[str, float] = {}

        for stem_name, stem_path_str in manifest.stems.items():
            est_p = Path(stem_path_str)
            gt_p = gt_dir / f"{stem_name}.wav"
            if est_p.is_file() and gt_p.is_file():
                y_est, _ = decode_audio(est_p, target_sr=sr, normalize_db=None)
                if y_est.ndim > 1:
                    y_est = np.mean(y_est, axis=1)
                y_gt, _ = decode_audio(gt_p, target_sr=sr, normalize_db=None)
                if y_gt.ndim > 1:
                    y_gt = np.mean(y_gt, axis=1)

                sdr_by_stem[stem_name] = compute_sdr(y_gt, y_est)

        avg_sdr = round(float(np.mean(list(sdr_by_stem.values()))), 2) if sdr_by_stem else 0.0
        separation_metrics = {
            "mode": "objective_ground_truth",
            "sdr": sdr_by_stem,
            "average_sdr_db": avg_sdr,
        }
    else:
        # Diagnostic heuristic: sum of separated stems vs mix residual energy
        stem_signals = []
        for stem_name, stem_path_str in manifest.stems.items():
            stem_p = Path(stem_path_str)
            if stem_p.is_file():
                y_s, _ = decode_audio(stem_p, target_sr=sr, normalize_db=None)
                if y_s.ndim > 1:
                    y_s = np.mean(y_s, axis=1)
                stem_signals.append(y_s)

        if stem_signals:
            min_len = min(len(y_mix), min(len(s) for s in stem_signals))
            y_sum = np.sum([s[:min_len] for s in stem_signals], axis=0)
            residual = y_mix[:min_len] - y_sum
            mix_energy = np.sum(y_mix[:min_len]**2)
            res_energy = np.sum(residual**2)
            residual_ratio_db = round(10.0 * np.log10(res_energy / (mix_energy + 1e-12)), 2)
        else:
            residual_ratio_db = 0.0

        separation_metrics = {
            "mode": "diagnostic_reconstruction",
            "stems_count": len(manifest.stems),
            "residual_reconstruction_db": residual_ratio_db,
        }

    # 3. Transcription Quality (Objective vs Diagnostic)
    if ground_truth_midi and Path(ground_truth_midi).is_file():
        is_objective = True
        try:
            gt_pm = pretty_midi.PrettyMIDI(str(ground_truth_midi))
            gt_onsets = sorted([n.start for inst in gt_pm.instruments for n in inst.notes])

            est_onsets = []
            for stem_mid in manifest.midis.values():
                m_p = Path(stem_mid)
                if m_p.is_file() and m_p.name != "all_stems.mid":
                    pm = pretty_midi.PrettyMIDI(str(m_p))
                    est_onsets.extend([n.start for inst in pm.instruments for n in inst.notes])
            est_onsets.sort()

            # Onset matching with 50ms tolerance
            tolerance = 0.05
            tp = 0
            gt_matched = set()
            for est_t in est_onsets:
                for idx, gt_t in enumerate(gt_onsets):
                    if idx not in gt_matched and abs(est_t - gt_t) <= tolerance:
                        tp += 1
                        gt_matched.add(idx)
                        break

            precision = round(tp / max(1, len(est_onsets)), 3)
            recall = round(tp / max(1, len(gt_onsets)), 3)
            f1 = round(2 * precision * recall / max(1e-6, precision + recall), 3)

            transcription_metrics = {
                "mode": "objective_ground_truth",
                "onset_precision": precision,
                "onset_recall": recall,
                "onset_f1": f1,
            }
        except Exception as e:
            transcription_metrics = {"error": str(e)}
    else:
        # Diagnostic note statistics
        stem_note_stats: Dict[str, Any] = {}
        total_notes = 0
        for stem_name, mid_path in manifest.midis.items():
            m_p = Path(mid_path)
            if m_p.is_file():
                try:
                    pm = pretty_midi.PrettyMIDI(str(m_p))
                    notes = [n for inst in pm.instruments for n in inst.notes]
                    total_notes += len(notes)
                    pitches = [n.pitch for n in notes]
                    stem_note_stats[stem_name] = {
                        "notes_count": len(notes),
                        "pitch_min": min(pitches) if pitches else None,
                        "pitch_max": max(pitches) if pitches else None,
                    }
                except Exception:
                    pass

        transcription_metrics = {
            "mode": "diagnostic_note_density",
            "total_notes": total_notes,
            "stems": stem_note_stats,
        }

    return QualityReport(
        is_objective=is_objective,
        separation_metrics=separation_metrics,
        transcription_metrics=transcription_metrics,
        audio_diagnostics=audio_diagnostics,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )
