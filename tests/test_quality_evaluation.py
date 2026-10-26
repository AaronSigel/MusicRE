from pathlib import Path
import numpy as np
import pretty_midi
import soundfile as sf
import pytest

from music_re.schemas import QualityReport
from music_re.core.quality import evaluate_quality
from music_re.workspace import WorkspaceManager

@pytest.fixture
def sample_workspace_for_quality(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_wav = tmp_path / "song.wav"
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    sf.write(str(track_wav), y, sr)

    stems_dir = ws.get_stems_dir(track_wav, model_name="htdemucs")
    for s in ["vocals", "drums", "bass", "other"]:
        sf.write(str(stems_dir / f"{s}.wav"), y * 0.25, sr)
    (stems_dir / ".complete_htdemucs").write_text("ok")

    midis_dir = ws.get_track_dir(track_wav) / "midis"
    midis_dir.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    pm.write(str(midis_dir / "vocals.mid"))
    ws.mark_stage_complete(track_wav, 'separation', {s: str(stems_dir / f'{s}.wav') for s in ['vocals', 'drums', 'bass', 'other']})
    ws.mark_stage_complete(track_wav, 'transcription', {'vocals': str(midis_dir / 'vocals.mid')})

    return ws, track_wav

def test_quality_evaluation_without_ground_truth(sample_workspace_for_quality):
    ws, track_wav = sample_workspace_for_quality

    report = evaluate_quality(ws, track_wav, ground_truth_stems=None, ground_truth_midi=None)
    assert isinstance(report, QualityReport)
    assert report.is_objective is False
    assert "audio_diagnostics" in report.model_dump()
    assert "separation_metrics" in report.model_dump()
    assert "transcription_metrics" in report.model_dump()
    assert report.audio_diagnostics["peak_dbfs"] <= 0.0

    # Save to workspace
    saved = ws.save_quality_report(track_wav, report)
    assert saved.exists()
    assert saved.name == "quality_report.json"

    manifest = ws.get_manifest(track_wav)
    assert "evaluation" in manifest.completed_stages

def test_quality_evaluation_with_ground_truth(tmp_path: Path, sample_workspace_for_quality):
    ws, track_wav = sample_workspace_for_quality

    # Ground truth stems
    gt_dir = tmp_path / "gt_stems"
    gt_dir.mkdir(parents=True, exist_ok=True)
    sr = 22050
    y = 0.5 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32)
    rng = np.random.default_rng(10)
    for s in ["vocals", "drums", "bass", "other"]:
        # Non-identical, non-silent references yield finite BSSEval scores.
        sf.write(str(gt_dir / f"{s}.wav"), y * 0.25 + rng.normal(0, .01, len(y)), sr)

    report = evaluate_quality(ws, track_wav, ground_truth_stems=gt_dir)
    assert report.is_objective is True
    assert "sdr" in report.separation_metrics
