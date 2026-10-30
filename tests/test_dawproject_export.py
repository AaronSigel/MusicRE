import zipfile
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from dawproject import ContentType, DawProject
from typer.testing import CliRunner

from music_re.cli import app
from music_re.core.dawproject_export import export_dawproject
from music_re.core.export import export_bundle
from music_re.schemas import MusicalTimeline, TimelineSection
from music_re.workspace import WorkspaceManager


@pytest.fixture
def daw_workspace(tmp_path: Path):
    """Build a synthetic workspace with stems, MIDI files, and timeline."""
    ws = WorkspaceManager(tmp_path / "cache")
    source = tmp_path / "test_track.wav"
    sr = 22050
    sf.write(source, np.zeros(sr * 2, dtype=np.float32), sr)

    stems = {}
    stems_dir = ws.get_stems_dir(source, "fixture")
    for name in ["vocals", "bass", "drums", "other"]:
        wav_file = stems_dir / f"{name}.wav"
        data = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32) * 0.1
        sf.write(wav_file, data, sr)
        stems[name] = str(wav_file)

    root = ws.get_track_dir(source)
    midis_dir = root / "midis"
    midis_dir.mkdir(parents=True, exist_ok=True)
    midis = {}
    results = {}

    for name, pitch in [("vocals", 60), ("bass", 36), ("drums", 38), ("other", 67)]:
        pm = pretty_midi.PrettyMIDI(initial_tempo=126.0)
        inst = pretty_midi.Instrument(program=0, is_drum=(name == "drums"))
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=0.25, end=0.75)
        )
        inst.notes.append(
            pretty_midi.Note(velocity=110, pitch=pitch + 2, start=1.0, end=1.5)
        )
        pm.instruments.append(inst)

        midi_file = midis_dir / f"{name}.mid"
        pm.write(str(midi_file))
        raw_file = midis_dir / f"{name}_raw.mid"
        pm.write(str(raw_file))

        midis[name] = str(midi_file)
        results[name] = dict(
            status="notes",
            note_count=len(inst.notes),
            backend="fixture",
            midi_path=str(midi_file),
            raw_midi_path=str(raw_file),
        )

    all_pm = pretty_midi.PrettyMIDI(initial_tempo=126.0)
    all_mid = midis_dir / "all_stems.mid"
    all_pm.write(str(all_mid))
    midis["all_stems"] = str(all_mid)

    timeline = MusicalTimeline(
        duration_sec=2.0,
        bpm=126.0,
        beats=[0.0, 0.476, 0.952, 1.428, 1.904],
        downbeats=[0.0, 1.904],
        sections=[
            TimelineSection(label="intro", start_sec=0.0, end_sec=1.0),
            TimelineSection(label="verse", start_sec=1.0, end_sec=2.0),
        ],
    )

    ws.mark_stage_complete(source, "separation", stems)
    ws.save_metadata(source, {"title": "Test Song", "artist": "MusicRE Artist", "bpm": 126.0})
    ws.mark_stage_complete(source, "analysis", {"bpm": 126.0})
    ws.save_timeline(source, timeline)
    ws.mark_stage_complete(source, "transcription", midis)

    from music_re.core.quality import evaluate_quality
    ws.save_quality_report(source, evaluate_quality(ws, source))

    ws.update_manifest(
        source,
        status="completed",
        transcription_results=results,
        run_config={"profile": "balanced", "six_stems": False},
    )

    return ws, source


def test_export_dawproject_creates_valid_file(daw_workspace, tmp_path: Path):
    ws, source = daw_workspace
    out_file = tmp_path / "exported.dawproject"

    res = export_dawproject(ws, source, out_file)
    assert res == out_file
    assert out_file.is_file()
    assert zipfile.is_zipfile(out_file)

    with zipfile.ZipFile(out_file, "r") as zf:
        names = zf.namelist()
        assert DawProject.PROJECT_FILE in names
        assert DawProject.METADATA_FILE in names
        assert "audio/vocals.wav" in names
        assert "audio/bass.wav" in names
        assert "audio/drums.wav" in names
        assert "audio/other.wav" in names

    project = DawProject.load_project(str(out_file))
    assert project.application.name == "MusicRE"
    assert project.application.version == "0.1.0"

    assert project.transport is not None
    assert project.transport.tempo.value == 126.0
    assert project.transport.time_signature.numerator == 4
    assert project.transport.time_signature.denominator == 4

    assert len(project.structure) >= 8

    audio_tracks = [t for t in project.structure if ContentType.AUDIO in t.content_type]
    note_tracks = [t for t in project.structure if ContentType.NOTES in t.content_type]
    assert len(audio_tracks) == 4
    assert len(note_tracks) >= 4

    assert project.arrangement is not None
    assert project.arrangement.markers is not None
    marker_names = [m.name for m in project.arrangement.markers.markers]
    assert marker_names == ["intro", "verse"]

    notes_found = False
    for lane in project.arrangement.lanes.lanes:
        for clip in lane.clips:
            if hasattr(clip.content, "notes") and clip.content.notes:
                notes_found = True
                assert len(clip.content.notes) == 2
                assert clip.content.notes[0].time == pytest.approx(0.25, abs=0.01)
                assert clip.content.notes[0].vel == 100.0
    assert notes_found

    metadata = DawProject.load_metadata(str(out_file))
    assert metadata.title == "Test Song"
    assert metadata.artist == "MusicRE Artist"

    stream = DawProject.stream_embedded(str(out_file), "audio/vocals.wav")
    assert len(stream.read()) > 0


def test_export_dawproject_fallback_timeline_and_metadata(tmp_path: Path):
    ws = WorkspaceManager(tmp_path / "cache")
    source = tmp_path / "blank.wav"
    sr = 22050
    sf.write(source, np.zeros(sr, dtype=np.float32), sr)

    stems = {}
    stems_dir = ws.get_stems_dir(source, "fixture")
    v_stem = stems_dir / "vocals.wav"
    sf.write(v_stem, np.zeros(sr, dtype=np.float32), sr)
    stems["vocals"] = str(v_stem)

    ws.mark_stage_complete(source, "separation", stems)
    ws.update_manifest(source, status="completed", run_config={})

    out_file = tmp_path / "blank.dawproject"
    res = export_dawproject(ws, source, out_file)
    assert res.is_file()

    project = DawProject.load_project(str(out_file))
    assert project.transport.tempo.value == 120.0
    assert project.transport.time_signature.numerator == 4
    assert project.arrangement.markers is None

    meta = DawProject.load_metadata(str(out_file))
    assert meta.title == "blank"


def test_export_bundle_with_dawproject_flag(daw_workspace, tmp_path: Path):
    ws, source = daw_workspace
    bundle_dir = tmp_path / "exported_bundle"

    res = export_bundle(ws, source, bundle_dir, export_dawproject=True)
    assert res.is_dir()
    daw_file = res / f"{source.stem}.dawproject"
    assert daw_file.is_file()
    assert zipfile.is_zipfile(daw_file)


def test_export_bundle_archive_with_dawproject_flag(daw_workspace, tmp_path: Path):
    ws, source = daw_workspace
    bundle_zip = tmp_path / "exported_bundle.zip"

    res = export_bundle(ws, source, bundle_zip, archive=True, export_dawproject=True)
    assert res.is_file()
    with zipfile.ZipFile(res, "r") as zf:
        names = zf.namelist()
        assert any(n.endswith(".dawproject") for n in names)


def test_export_bundle_direct_dawproject_target(daw_workspace, tmp_path: Path):
    ws, source = daw_workspace
    target = tmp_path / "direct.dawproject"

    res = export_bundle(ws, source, target, export_dawproject=True)
    assert res == target
    assert target.is_file()
    assert zipfile.is_zipfile(target)


def test_cli_export_dawproject_flag(daw_workspace, tmp_path: Path):
    ws, source = daw_workspace
    runner = CliRunner()
    target = tmp_path / "cli_out"

    res = runner.invoke(
        app,
        [
            "export",
            str(source),
            "-o",
            str(target),
            "-d",
            "--cache-root",
            str(ws.cache_root),
        ],
    )
    assert res.exit_code == 0, res.output
    assert (target / f"{source.stem}.dawproject").is_file()
