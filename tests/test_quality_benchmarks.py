import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from music_re.schemas import QualityReport
from scripts.run_quality_benchmarks import (
    _fresh_cache_root, _peak_rss_mb, prepare_all_tracks, render_markdown_report, run_reference_benchmarks,
)


def test_peak_rss_mb_reads_real_process_memory_not_python_heap(monkeypatch):
    # Review I2: tracemalloc only sees Python/numpy heap allocations, not
    # torch/ONNX memory (measured ~7x under real process RSS). Report the
    # OS-level resident set size instead.
    import resource as resource_module

    from types import SimpleNamespace
    monkeypatch.setattr("scripts.run_quality_benchmarks.resource.getrusage",
                         lambda who: SimpleNamespace(ru_maxrss=512000))
    monkeypatch.setattr("scripts.run_quality_benchmarks.sys.platform", "linux")

    assert _peak_rss_mb() == pytest.approx(500.0)  # 512000 KB on Linux


def test_prepare_all_tracks_isolates_one_preparation_failure(monkeypatch, tmp_path):
    # Review I1: prepare_track() failures (corrupt metadata.yaml, a missing
    # stem wav, an alignment check) ran unguarded in main(), aborting the
    # whole benchmark before run_reference_benchmarks ever saw any track.
    track_dirs = [tmp_path / "Track00001", tmp_path / "Track00002"]
    for d in track_dirs:
        d.mkdir()

    def fake_prepare_track(track_dir, out_dir):
        if track_dir.name == "Track00002":
            raise ValueError("misaligned")
        return {"track_id": track_dir.name, "group_note_counts": {}}

    monkeypatch.setattr("scripts.run_quality_benchmarks.prepare_track", fake_prepare_track)

    prepared, failures = prepare_all_tracks(track_dirs, tmp_path / "prepared")

    assert [t["track_id"] for t in prepared] == ["Track00001"]
    assert len(failures) == 1
    assert failures[0]["track_id"] == "Track00002"
    assert failures[0]["error"] == "misaligned"
    assert failures[0]["quality_report"] is None


def test_fresh_cache_root_is_unique_each_call(tmp_path):
    # Review C3: a fixed cache_root lets a second main() run resume stale
    # cached stems/MIDI from a previous (possibly different-code) run,
    # making history snapshots incomparable across versions.
    first = _fresh_cache_root(tmp_path)
    second = _fresh_cache_root(tmp_path)
    assert first != second
    assert first.parent == tmp_path and second.parent == tmp_path


def test_run_reference_benchmarks_isolates_one_failure(monkeypatch, tmp_path):
    prepared = [
        {"track_id": "Track00001", "mix_path": str(tmp_path / "Track00001" / "mix.wav"),
         "ground_truth_stems_dir": str(tmp_path / "Track00001" / "gt_stems"),
         "ground_truth_midi_path": str(tmp_path / "Track00001" / "gt.mid"),
         "group_note_counts": {"vocals": 0, "drums": 3, "bass": 2, "other": 4}},
        {"track_id": "Track00002", "mix_path": str(tmp_path / "Track00002" / "mix.wav"),
         "ground_truth_stems_dir": str(tmp_path / "Track00002" / "gt_stems"),
         "ground_truth_midi_path": str(tmp_path / "Track00002" / "gt.mid"),
         "group_note_counts": {"vocals": 0, "drums": 1, "bass": 1, "other": 1}},
    ]

    def fake_run_pipeline(file_path, **kwargs):
        if "Track00002" in str(file_path):
            raise RuntimeError("boom")
        return {"workspace_dir": str(tmp_path)}

    fake_report = QualityReport(is_objective=True, evaluated_at="now",
                                 transcription_metrics={"stems": {"vocals": {"notes_count": 7}}})
    monkeypatch.setattr("scripts.run_quality_benchmarks.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("scripts.run_quality_benchmarks.evaluate_quality", lambda *a, **k: fake_report)

    results = run_reference_benchmarks(prepared, cache_root=tmp_path)

    assert results[0]["error"] is None and results[0]["quality_report"] is not None
    assert results[0]["false_notes_on_absent_instrument"] == {"vocals": 7}  # ref had 0 vocal notes, estimate found 7
    assert results[1]["error"] == "boom" and results[1]["quality_report"] is None
    assert results[1]["false_notes_on_absent_instrument"] == {}


def test_render_markdown_report_lists_every_track():
    report = {"reference_tracks": [{"track_id": "Track00001",
                                     "quality_report": {"separation_metrics": {}, "transcription_metrics": {}},
                                     "wall_time_sec": 1.2, "peak_ram_mb": 300.0, "error": None,
                                     "false_notes_on_absent_instrument": {}}],
              "manual_spotcheck": [], "git_revision": "bc5ae18"}
    md = render_markdown_report(report)
    assert "Track00001" in md and "bc5ae18" in md


def test_render_markdown_report_shows_per_stem_sdr_and_drum_class_counts():
    # An average SDR hides whether every stem separated equally well, and a
    # 0.0 drum-class F1 is ambiguous without the raw ref/est note counts
    # behind it (real user feedback on a real benchmark run).
    report = {"reference_tracks": [{
        "track_id": "Track00001",
        "quality_report": {
            "separation_metrics": {"average_sdr_db": 14.625,
                                    "sdr": {"drums": 14.397, "bass": 13.113, "other": 16.365}},
            "transcription_metrics": {"drums_by_class": {
                "snare": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ref_count": 1, "est_count": 0},
                "hihat": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ref_count": 0, "est_count": 1},
            }},
        },
        "wall_time_sec": 1.2, "peak_ram_mb": 300.0, "error": None,
        "false_notes_on_absent_instrument": {},
    }], "manual_spotcheck": [], "git_revision": "bc5ae18"}

    md = render_markdown_report(report)

    assert "## Per-stem separation (SDR)" in md
    assert "drums" in md and "14.397" in md
    assert "bass" in md and "13.113" in md
    assert "other" in md and "16.365" in md
    assert "ref=1, est=0" in md  # snare: the model found nothing, not "wrong but present"
    assert "ref=0, est=1" in md  # hihat: a pure false positive, not a miss on a real event
