import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_quality_benchmarks import run_reference_benchmarks
from scripts.slakh_mapping import prepare_track
from tests.test_slakh_mapping import _write_fake_slakh_track


@pytest.mark.models
def test_benchmark_one_synthetic_track_end_to_end(tmp_path):
    track_dir = _write_fake_slakh_track(tmp_path / "Track00001", stems=[
        ("S00", "Guitar", False, 48),
        ("S01", "Drums", True, 0),
        ("S02", "Electric Bass", False, 33),
    ])
    prepared = [prepare_track(track_dir, tmp_path / "prepared")]

    results = run_reference_benchmarks(prepared, cache_root=tmp_path / "cache")

    assert results[0]["error"] is None
    assert results[0]["quality_report"]["is_objective"] is True
