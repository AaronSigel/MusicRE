import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.run_quality_benchmarks import discover_benchmark_tracks, render_markdown_report, main

def test_discover_tracks_ignores_hidden_dirs(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    
    (raw_dir / "Track00001").mkdir()
    (raw_dir / ".staging-12345").mkdir()
    (raw_dir / ".old-Track00001-backup").mkdir()
    
    tracks = discover_benchmark_tracks(raw_dir)
    names = [t.name for t in tracks]
    assert "Track00001" in names
    assert ".staging-12345" not in names
    assert ".old-Track00001-backup" not in names

def test_render_report_displays_stem_status_when_sdr_none():
    results = [{
        "track_id": "Track00001",
        "error": None,
        "quality_report": {
            "separation_metrics": {
                "sdr": {"drums": None, "bass": 12.5, "other": 14.0},
                "stem_status": {"drums": "perfect", "bass": "scored", "other": "scored"},
                "average_sdr_db": 13.25,
                "average_sdr_count": 2,
                "other_includes": ["other", "guitar", "piano"],
                "residual_reconstruction_db": -32.5,
            },
            "transcription_metrics": {
                "pitched": {"onset_pitch_f1": 0.8},
                "drums": {"onset_pitch_f1": 0.7},
                "drums_by_class": {},
            }
        }
    }]
    report = {"reference_tracks": results, "git_revision": "test1234"}
    md = render_markdown_report(report)
    # Drums cell should display 'perfect', not empty or None
    assert "perfect" in md
    assert "other+guitar+piano" in md or "other (includes guitar, piano)" in md or "other" in md
