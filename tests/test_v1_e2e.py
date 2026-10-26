from pathlib import Path
import zipfile
import numpy as np
import pretty_midi
import soundfile as sf
import pytest

from music_re.core.audio_io import decode_audio
from music_re.core.backends import get_structure_analyzer, get_transcriber_backend
from music_re.core.export import export_bundle
from music_re.core.midi_project import build_combined_midi_project
from music_re.core.quality import evaluate_quality
from music_re.core.separator import separate_stems
from music_re.workspace import WorkspaceManager


def test_v1_full_pipeline_end_to_end(tmp_path: Path):
    sr = 22050
    duration_sec = 4.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    # 1. Generate multi-stem audio
    y_melody = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    y_bass = 0.4 * np.sin(2 * np.pi * 110.0 * t)
    y_kick = 0.5 * np.sin(2 * np.pi * 50.0 * t) * (np.sin(2 * np.pi * 2.0 * t) > 0.9)
    y_mix = (y_melody + y_bass + y_kick).astype(np.float32)

    input_audio = tmp_path / "song.wav"
    sf.write(str(input_audio), y_mix, sr)

    # Setup isolated workspace
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_dir = ws.get_track_dir(input_audio)

    # 2. Decode & Normalize
    decoded, dec_sr = decode_audio(input_audio, target_sr=sr, normalize_db=-1.0)
    assert dec_sr == sr
    assert 0.88 <= np.max(np.abs(decoded)) <= 0.90

    # 3. Separation (mock stems to verify contract without 20s Demucs run)
    stems_dir = ws.get_stems_dir(input_audio, model_name="htdemucs")
    mock_stems = {
        "vocals": stems_dir / "vocals.wav",
        "bass": stems_dir / "bass.wav",
        "drums": stems_dir / "drums.wav",
        "other": stems_dir / "other.wav",
    }
    sf.write(str(mock_stems["vocals"]), y_melody, sr)
    sf.write(str(mock_stems["bass"]), y_bass, sr)
    sf.write(str(mock_stems["drums"]), y_kick, sr)
    sf.write(str(mock_stems["other"]), y_melody * 0.1, sr)
    (stems_dir / ".complete_htdemucs").write_text("ok")

    sep_res = separate_stems(input_audio, workspace=ws, six_stems=False)
    assert len(sep_res.stems) == 4
    ws.mark_stage_complete(input_audio, "separation", artifacts=sep_res.stems)

    # 4. Structure Analysis (timeline.json)
    structure_analyzer = get_structure_analyzer(name="librosa")
    timeline = structure_analyzer.analyze(input_audio, stems=sep_res.stems)
    tl_file = ws.save_timeline(input_audio, timeline)
    assert tl_file.exists()
    assert len(timeline.beats) > 0
    assert len(timeline.downbeats) > 0
    assert len(timeline.sections) > 0

    # 5. MIDI Transcription with timeline snapping
    midis_dir = track_dir / "midis"
    midis_dir.mkdir(parents=True, exist_ok=True)
    stem_midis = {}
    for name, stem_p in sep_res.stems.items():
        out_mid = midis_dir / f"{name}.mid"
        mode = "drums" if name == "drums" else "tonal"
        transcriber = get_transcriber_backend(mode=mode)
        transcriber.transcribe(
            audio_path=stem_p,
            output_midi_path=out_mid,
            timeline=timeline,
            bpm=timeline.bpm,
            quantize=True,
            save_raw=True,
        )
        assert out_mid.exists()
        assert (midis_dir / f"{name}_raw.mid").exists(), "Raw unquantized copy must exist"
        stem_midis[name] = str(out_mid)

    # Combined project
    all_stems_path = midis_dir / "all_stems.mid"
    build_combined_midi_project(stem_midis, all_stems_path, timeline=timeline, bpm=timeline.bpm)
    assert all_stems_path.exists()
    stem_midis["all_stems"] = str(all_stems_path)
    ws.mark_stage_complete(input_audio, "transcription", artifacts=stem_midis)

    # 6. Quality Evaluation (quality_report.json)
    report = evaluate_quality(ws, input_audio)
    qr_file = ws.save_quality_report(input_audio, report)
    assert qr_file.exists()
    assert report.is_objective is False
    assert "audio_diagnostics" in report.model_dump()
    assert "separation_metrics" in report.model_dump()

    # 7. Final Bundle Export
    export_dir = tmp_path / "v1_release_bundle"
    export_bundle(ws, input_audio, export_dir, profile="quality", archive=False)
    assert (export_dir / "timeline.json").is_file()
    assert (export_dir / "manifest.json").is_file()
    assert (export_dir / "quality_report.json").is_file()
    assert (export_dir / "stems" / "vocals.wav").is_file()
    assert (export_dir / "stems" / "drums.wav").is_file()
    assert (export_dir / "stems" / "bass.wav").is_file()
    assert (export_dir / "stems" / "other.wav").is_file()
    assert (export_dir / "midis" / "all_stems.mid").is_file()
    assert (export_dir / "midis" / "vocals_raw.mid").is_file()

    # Export ZIP
    zip_target = tmp_path / "v1_release.zip"
    export_bundle(ws, input_audio, zip_target, archive=True)
    assert zip_target.is_file()
    with zipfile.ZipFile(str(zip_target), "r") as zf:
        names = zf.namelist()
        assert any("all_stems.mid" in n for n in names)
        assert any("timeline.json" in n for n in names)
        assert any("quality_report.json" in n for n in names)
