import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from music_re.core.quality import QualityReport
from scripts.run_quality_benchmarks import (
    prepare_all_tracks, render_markdown_report, run_reference_benchmarks,
)
from scripts.slakh_mapping import classify_stem, prepare_track

SR = 22050


def _write_fake_slakh_track(track_dir: Path, stems, duration_sec: float = 1.0, amplitude: float = 0.1):
    """Same layout as tests/test_slakh_mapping.py::_write_fake_slakh_track.

    stems: (stem_id, inst_class, is_drum, program_num, has_audio, has_midi)
    """
    track_dir.mkdir(parents=True, exist_ok=True)
    (track_dir / "MIDI").mkdir(exist_ok=True)
    (track_dir / "stems").mkdir(exist_ok=True)

    n_samples = int(SR * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    mix = np.zeros(n_samples)
    metadata = {"stems": {}}

    for i, (stem_id, inst_class, is_drum, program_num, has_audio, has_midi) in enumerate(stems):
        # Real BabySlakh metadata marks every stem audio_rendered/midi_saved
        # False even though the files ship, so the flags carry no signal.
        metadata["stems"][stem_id] = dict(
            inst_class=inst_class, is_drum=is_drum, midi_program_name=inst_class,
            program_num=program_num, audio_rendered=False, midi_saved=False,
        )
        if has_midi:
            pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            inst = pretty_midi.Instrument(program=program_num, is_drum=is_drum)
            inst.notes.append(pretty_midi.Note(velocity=100, pitch=36 if is_drum else 60,
                                                start=0.1 * (i + 1), end=0.1 * (i + 1) + 0.2))
            pm.instruments.append(inst)
            pm.write(str(track_dir / "MIDI" / f"{stem_id}.mid"))
        if has_audio:
            wave = amplitude * np.sin(2 * np.pi * 220.0 * (i + 1) * t)
            sf.write(str(track_dir / "stems" / f"{stem_id}.wav"), wave, SR, subtype="FLOAT")
            mix = mix + wave

    (track_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    sf.write(str(track_dir / "mix.wav"), mix, SR, subtype="FLOAT")
    return track_dir


def test_audio_without_midi_is_missing_annotation_not_absent(tmp_path, monkeypatch):
    track_dir = _write_fake_slakh_track(tmp_path / "raw" / "Track00001", stems=[
        ("S00", "Guitar", False, 30, True, True),
        ("S01", "Drums", True, 0, True, True),
        ("S02", "Bass", False, 33, True, False),  # audio, no MIDI
    ])
    prepared, failures = prepare_all_tracks([track_dir], tmp_path / "prepared")
    assert not failures
    track = prepared[0]
    assert track["group_status"] == {
        "vocals": "absent", "drums": "present", "bass": "missing_annotation", "other": "present",
    }
    assert track["group_note_counts"]["bass"] == 0
    assert any("S02" in w for w in track["warnings"])

    fake_report = QualityReport(is_objective=True, evaluated_at="now", transcription_metrics={
        "stems": {"vocals": {"notes_count": 4}, "bass": {"notes_count": 9}}})
    monkeypatch.setattr("scripts.run_quality_benchmarks.run_pipeline", lambda *a, **k: {})
    monkeypatch.setattr("scripts.run_quality_benchmarks.evaluate_quality", lambda *a, **k: fake_report)

    results = run_reference_benchmarks(prepared, cache_root=tmp_path / "cache")
    entry = results[0]
    assert entry["false_notes_on_absent_instrument"] == {"vocals": 4}
    assert entry["group_status"]["bass"] == "missing_annotation"
    assert any("S02" in w for w in entry["preparation_warnings"])

    md = render_markdown_report({"reference_tracks": results, "manual_spotcheck": [], "git_revision": "x"})
    assert "missing_annotation" in md
    assert "S02" in md


def test_midi_without_audio_is_not_rendered(tmp_path):
    track_dir = _write_fake_slakh_track(tmp_path / "Track00002", stems=[
        ("S00", "Piano", False, 0, True, True),
        ("S01", "Drums", True, 0, False, True),
    ])
    result = prepare_track(track_dir, tmp_path / "prepared" / "Track00002")
    assert result["group_status"]["drums"] == "not_rendered"
    assert result["group_status"]["bass"] == "absent"


def test_stem_with_neither_audio_nor_midi_is_excluded_from_status(tmp_path):
    # Real BabySlakh lists stems (e.g. Track00001/S06) that ship neither a
    # wav nor a MIDI file; they are in neither the mix nor the annotation.
    track_dir = _write_fake_slakh_track(tmp_path / "Track00008", stems=[
        ("S00", "Guitar", False, 30, True, True),
        ("S01", "Organ", False, 16, False, False),
        ("S02", "Bass", False, 33, False, False),
    ])
    result = prepare_track(track_dir, tmp_path / "prepared" / "Track00008")
    assert result["group_status"]["other"] == "present"
    assert result["group_status"]["bass"] == "absent"
    assert any("S01" in w and "excluded" in w for w in result["warnings"])
    assert any("S02" in w and "excluded" in w for w in result["warnings"])


def test_status_without_group_status_never_counts_absent(monkeypatch, tmp_path):
    # A track entry with no completeness info must not be treated as
    # "every zero-note group is absent".
    prepared = [{"track_id": "T", "mix_path": "m.wav", "ground_truth_stems_dir": "s",
                 "ground_truth_midi_path": "g.mid",
                 "group_note_counts": {"vocals": 0, "drums": 1, "bass": 0, "other": 1}}]
    fake_report = QualityReport(is_objective=True, evaluated_at="now", transcription_metrics={
        "stems": {"vocals": {"notes_count": 4}}})
    monkeypatch.setattr("scripts.run_quality_benchmarks.run_pipeline", lambda *a, **k: {})
    monkeypatch.setattr("scripts.run_quality_benchmarks.evaluate_quality", lambda *a, **k: fake_report)
    results = run_reference_benchmarks(prepared, cache_root=tmp_path)
    assert results[0]["false_notes_on_absent_instrument"] == {}


def test_exact_inst_class_taxonomy():
    assert classify_stem("Bass", is_drum=False) == "bass"
    assert classify_stem("Bassoon", is_drum=False) == "other"
    assert classify_stem("Reed", is_drum=False) == "other"
    assert classify_stem("Drums", is_drum=False) == "drums"
    assert classify_stem("Piano", is_drum=False) == "other"
    assert classify_stem("Guitar", is_drum=False) == "other"
    assert classify_stem("Strings (continued)", is_drum=False) == "other"
    assert classify_stem("Percussive", is_drum=True) == "drums"


def test_bassoon_stem_lands_in_other(tmp_path):
    track_dir = _write_fake_slakh_track(tmp_path / "Track00004", stems=[
        ("S00", "Bassoon", False, 70, True, True),
    ])
    result = prepare_track(track_dir, tmp_path / "prepared" / "Track00004")
    gt = Path(result["ground_truth_stems_dir"])
    assert (gt / "other.wav").exists()
    assert not (gt / "bass.wav").exists()
    assert result["group_status"]["bass"] == "absent"
    assert result["group_status"]["other"] == "present"


def test_reprepare_removes_stale_group_wav(tmp_path):
    out_dir = tmp_path / "prepared" / "Track00005"
    first = _write_fake_slakh_track(tmp_path / "v1" / "Track00005", stems=[
        ("S00", "Guitar", False, 30, True, True),
        ("S01", "Bass", False, 33, True, True),
    ])
    prepare_track(first, out_dir)
    assert (out_dir / "ground_truth_stems" / "bass.wav").exists()
    (out_dir / "ground_truth_stems" / "vocals.wav").write_bytes(b"stale")

    second = _write_fake_slakh_track(tmp_path / "v2" / "Track00005", stems=[
        ("S00", "Guitar", False, 30, True, True),
    ])
    result = prepare_track(second, out_dir)

    gt = Path(result["ground_truth_stems_dir"])
    assert gt == out_dir.resolve() / "ground_truth_stems"
    assert not (gt / "vocals.wav").exists()
    assert not (gt / "bass.wav").exists()
    assert (gt / "other.wav").exists()
    assert sorted(p.name for p in out_dir.parent.iterdir()) == ["Track00005"]


def test_failed_reprepare_keeps_previous_prepared_dir(tmp_path):
    out_dir = tmp_path / "prepared" / "Track00006"
    good = _write_fake_slakh_track(tmp_path / "v1" / "Track00006", stems=[
        ("S00", "Guitar", False, 30, True, True),
    ])
    prepare_track(good, out_dir)
    before = (out_dir / "ground_truth_stems" / "other.wav").read_bytes()

    bad = _write_fake_slakh_track(tmp_path / "v2" / "Track00006", stems=[
        ("S00", "Guitar", False, 30, True, True),
    ])
    (bad / "metadata.yaml").write_text("stems: [unclosed")
    try:
        prepare_track(bad, out_dir)
    except Exception:
        pass
    assert (out_dir / "ground_truth_stems" / "other.wav").read_bytes() == before
    assert sorted(p.name for p in out_dir.parent.iterdir()) == ["Track00006"]


def test_group_sum_above_one_is_not_clipped(tmp_path):
    track_dir = _write_fake_slakh_track(tmp_path / "Track00007", stems=[
        ("S00", "Bass", False, 33, True, True),
        ("S01", "Bass", False, 34, True, True),
    ], amplitude=0.8)
    # Same waveform for both stems so the group sum peaks at 1.6.
    wave, _ = sf.read(str(track_dir / "stems" / "S00.wav"))
    sf.write(str(track_dir / "stems" / "S01.wav"), wave, SR, subtype="FLOAT")

    result = prepare_track(track_dir, tmp_path / "prepared" / "Track00007")
    bass_path = Path(result["ground_truth_stems_dir"]) / "bass.wav"
    assert sf.info(str(bass_path)).subtype == "FLOAT"
    data, _ = sf.read(str(bass_path))
    assert np.max(np.abs(data)) > 1.55
