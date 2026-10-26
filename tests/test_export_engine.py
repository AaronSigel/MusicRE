from pathlib import Path
import zipfile
import pretty_midi
import soundfile as sf
import numpy as np
import pytest

from music_re.core.export import export_bundle
from music_re.schemas import MusicalTimeline, QualityReport, TimelineSection
from music_re.workspace import WorkspaceManager

@pytest.fixture
def populated_workspace(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_wav = tmp_path / "track.wav"
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    sf.write(str(track_wav), y, sr)

    # 1. Stems
    stems_dir = ws.get_stems_dir(track_wav, model_name="htdemucs")
    stems = {}
    for s in ["vocals", "drums", "bass", "other"]:
        sf.write(str(stems_dir / f"{s}.wav"), y * 0.25, sr)
        stems[s] = str(stems_dir / f"{s}.wav")
    (stems_dir / ".complete_htdemucs").write_text("ok")
    ws.mark_stage_complete(track_wav, "separation", artifacts=stems)

    # 2. Timeline
    tl = MusicalTimeline(
        duration_sec=1.0,
        bpm=120.0,
        beats=[0.0, 0.5, 1.0],
        downbeats=[0.0],
        sections=[TimelineSection(label="intro", start_sec=0.0, end_sec=1.0, confidence=1.0)],
    )
    ws.save_timeline(track_wav, tl)

    # 3. MIDIs
    midis_dir = ws.get_track_dir(track_wav) / "midis"
    midis_dir.mkdir(parents=True, exist_ok=True)
    midis = {}
    for s in ["vocals", "drums", "bass", "other", "all_stems"]:
        p = midis_dir / f"{s}.mid"
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        pm.instruments.append(pretty_midi.Instrument(program=0))
        pm.write(str(p))
        midis[s] = str(p)
    ws.mark_stage_complete(track_wav, "transcription", artifacts=midis)

    # 4. Quality Report
    qr = QualityReport(
        is_objective=False,
        separation_metrics={"residual_reconstruction_db": -24.5},
        transcription_metrics={"total_notes": 12},
        audio_diagnostics={"peak_dbfs": -1.0},
        evaluated_at="2026-10-26T12:00:00Z",
    )
    ws.save_quality_report(track_wav, qr)

    return ws, track_wav

def test_export_bundle_directory(tmp_path: Path, populated_workspace):
    ws, track_wav = populated_workspace
    export_out = tmp_path / "export_output"

    res = export_bundle(ws, track_wav, export_out, archive=False)
    assert res.is_dir()
    assert (res / "timeline.json").is_file()
    assert (res / "manifest.json").is_file()
    assert (res / "quality_report.json").is_file()
    assert (res / "stems" / "vocals.wav").is_file()
    assert (res / "midis" / "all_stems.mid").is_file()

def test_export_bundle_zip_archive(tmp_path: Path, populated_workspace):
    ws, track_wav = populated_workspace
    export_zip_target = tmp_path / "export_archive.zip"

    res = export_bundle(ws, track_wav, export_zip_target, archive=True)
    assert res.is_file()
    assert res.suffix == ".zip"

    with zipfile.ZipFile(str(res), "r") as zf:
        names = zf.namelist()
        assert any("timeline.json" in n for n in names)
        assert any("manifest.json" in n for n in names)
        assert any("quality_report.json" in n for n in names)
        assert any("all_stems.mid" in n for n in names)
