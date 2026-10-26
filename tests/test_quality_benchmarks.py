import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from music_re.schemas import QualityReport
from scripts.run_quality_benchmarks import render_markdown_report, run_reference_benchmarks


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
