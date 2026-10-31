import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import calibrate_spectral as cs  # noqa: E402
from scripts.run_quality_benchmarks import track_in_split  # noqa: E402
from music_re.core.midi_validation import SCORE_FEATURES  # noqa: E402

TUNE = [t for t in (f'Track{i:05d}' for i in range(1, 30)) if track_in_split(t, 'tune')][:3]
TEST = [t for t in (f'Track{i:05d}' for i in range(1, 30)) if track_in_split(t, 'test')][:2]


def _write_track(cache_root, prepared_dir, track, seed, invert=False, n=120):
    rng = np.random.default_rng(seed)
    ref = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=33, name='bass')
    notes = []
    for i in range(n):
        start = 0.5 * i
        is_tp = rng.random() < 0.7
        if is_tp:
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=40, start=start + 0.01, end=start + 0.4))
        good = is_tp != invert
        support = rng.uniform(0.3, 1.0) if good else rng.uniform(0.0, 0.5)
        notes.append(dict(pitch=40 if is_tp else 41, start_sec=start, end_sec=start + 0.4, velocity=90,
                          harmonic_support=support, spectral_contrast=rng.uniform(0, 1),
                          temporal_persistence=rng.uniform(0, 1), onset_agreement=rng.uniform(0, 1),
                          leakage=rng.uniform(0, 1), note_score=None,
                          decision='reject' if support < 0.05 else 'keep', reason=None))
    # One out-of-range note: never fitted, never removed by score.
    notes.append(dict(pitch=41, start_sec=999.0, end_sec=999.5, velocity=90, harmonic_support=0,
                      spectral_contrast=0, temporal_persistence=0, onset_agreement=0, leakage=0,
                      note_score=None, decision='reject', reason='out_of_range'))
    ref.instruments.append(inst)
    (prepared_dir / track).mkdir(parents=True)
    ref.write(str(prepared_dir / track / 'ground_truth.mid'))
    ws = cache_root / track / 'mix_abc'
    ws.mkdir(parents=True)
    report = {'mode': 'filter', 'algo_version': 'spectral-v1', 'config': {}, 'active_variant': 'validated',
              'stems': {'bass': {'stem': 'bass', 'mode': 'filter', 'status': 'validated', 'notes': notes,
                                 'counts': {}, 'warnings': [], 'input_midi_path': 'x',
                                 'output_midi_path': 'y'},
                        'drums': {'stem': 'drums', 'mode': 'filter', 'status': 'skipped', 'notes': [],
                                  'counts': {}, 'warnings': [], 'input_midi_path': 'x',
                                  'output_midi_path': 'y'}}}
    (ws / 'spectral_validation.json').write_text(json.dumps(report))


def _dataset(tmp_path, invert_test=False):
    cache_root, prepared = tmp_path / 'cache', tmp_path / 'prepared'
    for i, track in enumerate(TUNE):
        _write_track(cache_root, prepared, track, seed=i)
    for i, track in enumerate(TEST):
        _write_track(cache_root, prepared, track, seed=100 + i, invert=invert_test)
    return cache_root, prepared


def test_pick_threshold_is_highest_with_recall_loss_under_cap():
    scores = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
    labels = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    # 7 TPs: removing below .3 loses 0; below .4 loses 1/7 > cap.
    assert cs.pick_threshold(scores, labels, cap=0.02) == pytest.approx(.3)
    assert cs.pick_threshold(scores, labels, cap=0.15) == pytest.approx(.5)
    assert cs.pick_threshold(np.array([.5, .6]), np.array([1, 1]), cap=0.0) == pytest.approx(.5)


def test_labels_require_equal_pitch_and_onset_within_50ms():
    ref = {40: np.array([1.0])}
    assert cs.is_tp(ref, 40, 1.049)
    assert not cs.is_tp(ref, 40, 1.06)
    assert not cs.is_tp(ref, 41, 1.0)


def test_calibration_respects_cap_on_tune_and_ignores_test_split(tmp_path):
    cache_root, prepared = _dataset(tmp_path)
    result = cs.calibrate(cache_root, prepared, None, roles=('bass',), recall_cap=0.02)
    bass = result['roles']['bass']
    assert len(bass['weights']) == len(bass['mean']) == len(bass['scale']) == len(SCORE_FEATURES)
    assert bass['uncertain_below'] >= bass['reject_below']
    tune = bass['metrics']['tune']['model']
    assert tune['tp_kept'] >= 0.98
    assert tune['removed'] > 0
    assert tune['precision'] > bass['metrics']['tune']['unfiltered']['precision']
    assert bass['metrics']['tune']['n'] == 3 * 120  # out_of_range notes excluded from fitting rows
    # Raising the threshold to the next tune score would break the cap.
    rows = [r for r in cs.collect_rows(cache_root, prepared, None, ('bass',)) if r.split == 'tune' and r.fit]
    x = np.array([r.features for r in rows])
    y = np.array([r.label for r in rows])
    scores = cs.score(bass, x)
    higher = np.unique(scores[scores > bass['reject_below']])
    assert 1 - y[scores >= higher[0]].sum() / y.sum() > 0.02
    # Test split cannot influence fitted parameters.
    other_root, other_prep = _dataset(tmp_path / 'inverted', invert_test=True)
    flipped = cs.calibrate(other_root, other_prep, None, roles=('bass',), recall_cap=0.02)['roles']['bass']
    for key in ('weights', 'bias', 'mean', 'scale', 'reject_below', 'uncertain_below'):
        assert flipped[key] == bass[key]
    assert flipped['metrics']['test'] != bass['metrics']['test']


def test_runtime_score_matches_script_score(tmp_path):
    from music_re.core.midi_validation import RoleRule, note_score
    from music_re.schemas import NoteEvidence
    cache_root, prepared = _dataset(tmp_path)
    bass = cs.calibrate(cache_root, prepared, None, roles=('bass',), recall_cap=0.02)['roles']['bass']
    rule = RoleRule(True, .1, .2, .1, .2, **cs.rule_params(bass))
    item = NoteEvidence(pitch=40, start_sec=0, end_sec=.4, velocity=90, harmonic_support=.4,
                        spectral_contrast=.5, temporal_persistence=.6, onset_agreement=.7, leakage=.1)
    x = np.array([[.4, .5, .6, .7, .1, np.log(.4)]])
    assert note_score(rule, item) == pytest.approx(float(cs.score(bass, x)[0]))


def test_cli_writes_json_with_metrics_and_absolute_scores(tmp_path):
    cache_root, prepared = _dataset(tmp_path)
    out = tmp_path / 'calib' / 'out.json'
    assert cs.main(['--cache-root', str(cache_root), '--prepared-dir', str(prepared),
                    '--out', str(out), '--roles', 'bass']) == 0
    data = json.loads(out.read_text())
    bass = data['roles']['bass']
    for split in ('tune', 'test'):
        metrics = bass['metrics'][split]
        assert set(metrics) >= {'n', 'unfiltered', 'current_rule', 'model', 'auc', 'absolute'}
        assert set(metrics['absolute']) == {'unfiltered', 'current_rule', 'model'}
    assert {'recommend_update', 'recommend_reason'} <= set(bass)
    assert data['recall_cap'] == 0.02
    assert data['features'] == list(SCORE_FEATURES)
    assert {t['track'] for t in data['per_track']} == set(TUNE + TEST)


def test_missing_prepared_reference_is_built_from_raw(tmp_path, monkeypatch):
    cache_root, prepared = _dataset(tmp_path)
    track = TUNE[0]
    saved = (prepared / track / 'ground_truth.mid').read_bytes()
    (prepared / track / 'ground_truth.mid').unlink()
    (prepared / track).rmdir()
    calls = []

    def fake_prepare(track_dir, out_dir):
        calls.append((Path(track_dir), Path(out_dir)))
        Path(out_dir).mkdir(parents=True)
        (Path(out_dir) / 'ground_truth.mid').write_bytes(saved)
        return {'octave_shift': {'bass': -12}, 'warnings': []}

    monkeypatch.setattr(cs, 'prepare_track', fake_prepare)
    cs.collect_rows(cache_root, prepared, tmp_path / 'raw', ('bass',))
    assert calls == [(tmp_path / 'raw' / track, prepared / track)]
    assert json.loads((prepared / track / 'prepare_result.json').read_text())['octave_shift']['bass'] == -12
