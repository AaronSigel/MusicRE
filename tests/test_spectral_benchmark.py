import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import run_quality_benchmarks as bench


def test_split_is_deterministic_and_disjoint():
    ids = [f'track{i}' for i in range(100)]
    tune = [x for x in ids if bench.track_in_split(x, 'tune')]
    test = [x for x in ids if bench.track_in_split(x, 'test')]
    assert tune and test
    assert set(tune).isdisjoint(test)
    assert set(tune + test) == set(ids)
    assert tune == [x for x in ids if bench.track_in_split(x, 'tune')]


def test_comparison_reports_distribution_and_recall_boundary():
    def row(track, f1, recall):
        return {'track_id': track, 'quality_report': {'transcription_metrics': {'pitched': {
            'onset_pitch_f1': f1, 'onset_pitch_recall': recall}}}}
    base = [row('a', .8, .8), row('b', .8, .8)]
    trial = [row('a', .77, .78), row('b', .81, .78)]
    comparison = bench.compare_modes(base, trial)
    assert comparison['f1_distribution']['min'] == pytest.approx(-.03)
    assert comparison['f1_distribution']['median'] == pytest.approx(-.01)
    assert comparison['f1_distribution']['max'] == pytest.approx(.01)
    assert comparison['f1_distribution']['drops_over_0_02'] == 1
    assert comparison['acceptance']['recall']['status'] == 'PASS'
    trial = [row('a', .77, .779), row('b', .81, .779)]
    assert bench.compare_modes(base, trial)['acceptance']['recall']['status'] == 'FAIL'
