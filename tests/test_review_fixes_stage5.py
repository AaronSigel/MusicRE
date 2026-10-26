from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from music_re.core.export import export_bundle
from music_re.core.quality import evaluate_quality
from music_re.workspace import WorkspaceManager


def test_export_bundle_prevents_source_overwrite(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_wav = tmp_path / "song.wav"
    sr = 22050
    sf.write(str(track_wav), np.zeros(sr, dtype=np.float32), sr)

    # Attempting to export directly targeting the source file must raise ValueError
    with pytest.raises(ValueError, match="Target path cannot overwrite source input file"):
        export_bundle(ws, track_wav, track_wav)

    # Attempting to export directly into workspace directory must raise ValueError
    with pytest.raises(ValueError, match="Target path cannot be inside the track cache workspace"):
        export_bundle(ws, track_wav, ws.get_track_dir(track_wav))


def test_quality_report_includes_snr_estimate(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_wav = tmp_path / "song.wav"
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Signal with noise
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t) + np.random.uniform(-0.01, 0.01, sr)
    sf.write(str(track_wav), y.astype(np.float32), sr)

    report = evaluate_quality(ws, track_wav)
    assert "snr_estimate_db" in report.audio_diagnostics
    assert isinstance(report.audio_diagnostics["snr_estimate_db"], float)


def test_export_bundle_respects_profiles(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_wav = tmp_path / "song.wav"
    sr = 22050
    sf.write(str(track_wav), np.zeros(sr, dtype=np.float32), sr)

    # Populate mock stems and midis
    stems_dir = ws.get_stems_dir(track_wav, model_name="htdemucs")
    for s in ["vocals", "drums", "bass", "other"]:
        sf.write(str(stems_dir / f"{s}.wav"), np.zeros(sr, dtype=np.float32), sr)
    (stems_dir / ".complete_htdemucs").write_text("ok")
    ws.mark_stage_complete(track_wav, "separation", artifacts={"vocals": str(stems_dir / "vocals.wav")})

    midis_dir = ws.get_track_dir(track_wav) / "midis"
    midis_dir.mkdir(parents=True, exist_ok=True)
    (midis_dir / "vocals.mid").write_bytes(b"MThd")
    (midis_dir / "vocals_raw.mid").write_bytes(b"MThd_RAW")
    ws.mark_stage_complete(track_wav, "transcription", artifacts={"vocals": str(midis_dir / "vocals.mid")})

    # Under 'fast' profile, raw backup midis should be excluded from final export bundle
    fast_export = tmp_path / "export_fast"
    export_bundle(ws, track_wav, fast_export, profile="fast")
    assert (fast_export / "midis" / "vocals.mid").is_file()
    assert not (fast_export / "midis" / "vocals_raw.mid").exists()

    # Under 'quality' profile, raw backup midis should be included
    quality_export = tmp_path / "export_quality"
    export_bundle(ws, track_wav, quality_export, profile="quality")
    assert (quality_export / "midis" / "vocals_raw.mid").is_file()
