"""Quality evidence: reference scores and diagnostics are separate contracts."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np

from music_re.core.audio_io import decode_audio
from music_re.core.quality_audio import separation_quality
from music_re.core.quality_midi import transcription_quality
from music_re.schemas import QualityReport
from music_re.workspace import WorkspaceManager


def evaluate_quality(workspace: WorkspaceManager, file_path: Union[str, Path],
                     ground_truth_stems: Optional[Union[str, Path]] = None,
                     ground_truth_midi: Optional[Union[str, Path]] = None) -> QualityReport:
    """Evaluate registered active artifacts, never infer success from a path alone."""
    p = Path(file_path).resolve()
    y, sr = decode_audio(p, target_sr=22050, normalize_db=None)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    p10 = float(np.percentile(np.abs(y), 10)) if y.size else 0.0
    diagnostics = dict(duration_sec=len(y) / sr,
                       peak_dbfs=float(20 * np.log10(peak)) if peak > 0 else None,
                       clipping_samples=int(np.sum(np.abs(y) >= .999)),
                       rms_to_p10_amplitude_db=float(20 * np.log10(rms / p10)) if p10 > 0 else None)
    manifest = workspace.get_manifest(p)
    separation = separation_quality(y, sr, manifest.stems, ground_truth_stems)
    transcription = transcription_quality(manifest.midis, ground_truth_midi)
    return QualityReport(is_objective=bool(separation['comparisons'] or transcription['comparisons']),
                         separation_metrics=separation, transcription_metrics=transcription,
                         audio_diagnostics=diagnostics, evaluated_at=datetime.now(timezone.utc).isoformat())
