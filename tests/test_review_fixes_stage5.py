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


def test_quality_report_includes_amplitude_diagnostic(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_wav = tmp_path / "song.wav"
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Signal with noise
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t) + np.random.uniform(-0.01, 0.01, sr)
    sf.write(str(track_wav), y.astype(np.float32), sr)

    report = evaluate_quality(ws, track_wav)
    assert "rms_to_p10_amplitude_db" in report.audio_diagnostics
    assert isinstance(report.audio_diagnostics["rms_to_p10_amplitude_db"], float)


def test_export_bundle_respects_profiles(tmp_path: Path, complete_workspace):
    # Under 'fast' profile, raw backup midis should be excluded from final export bundle
    from music_re.core.profiles import resolve_profile
    ws, track_wav = complete_workspace
    ws.update_manifest(track_wav, run_config=resolve_profile('fast').to_dict())
    fast_export = tmp_path / "export_fast"
    export_bundle(ws, track_wav, fast_export, profile="fast")
    assert (fast_export / "midis" / "vocals.mid").is_file()
    assert not (fast_export / "midis" / "vocals_raw.mid").exists()

    # Under 'quality' profile, raw backup midis should be included
    quality_export = tmp_path / "export_quality"
    ws.update_manifest(track_wav, run_config=resolve_profile('quality').to_dict())
    export_bundle(ws, track_wav, quality_export, profile="quality")
    assert (quality_export / "midis" / "vocals_raw.mid").is_file()
