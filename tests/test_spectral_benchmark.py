import json
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


def _qr(f1=.8, precision=.8, recall=.8, stems=None, sdr=10.0):
    return {'separation_metrics': {'average_sdr_db': sdr, 'sdr': {'bass': sdr}},
            'transcription_metrics': {
                'pitched': {'onset_pitch_f1': f1, 'onset_pitch_precision': precision,
                            'onset_pitch_recall': recall},
                'stems': stems or {'bass': {'notes_count': 10}, 'vocals': {'notes_count': 10}}}}


def _run(time=10.0, rss=1000.0, per_stem=None, error=None, **kwargs):
    return {'quality_report': None if error else _qr(**kwargs), 'wall_time_sec': time,
            'peak_rss_mb': rss, 'error': error, 'per_stem_f1': per_stem or {'bass': .8}}


def _track(track_id, cold=None, off=None, **modes):
    return {'track_id': track_id, 'group_status': {'vocals': 'absent', 'bass': 'present'},
            'runs': {'off_cold': cold or _run(time=100.0), 'off': off or _run(), **modes}}


def test_cli_defaults_and_choices():
    args = bench.build_parser().parse_args([])
    assert args.spectral_validation == ['off'] and args.split == 'all'
    args = bench.build_parser().parse_args(['--spectral-validation', 'off', 'audit', 'filter', '--split', 'test'])
    assert args.spectral_validation == ['off', 'audit', 'filter'] and args.split == 'test'
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(['--spectral-validation', 'bogus'])


def test_ab_calls_worker_per_mode_with_same_cache_root(tmp_path):
    calls = []

    def runner(spec):
        calls.append(spec)
        return {'quality_report': _qr(stems={'vocals': {'notes_count': 4}}), 'wall_time_sec': 1.0,
                'peak_rss_mb': 100.0, 'error': None, 'per_stem_f1': {}}

    prepared = [{'track_id': t, 'mix_path': f'/m/{t}.wav', 'ground_truth_stems_dir': f'/g/{t}',
                 'ground_truth_midi_path': f'/g/{t}.mid', 'group_status': {'vocals': 'absent'}}
                for t in ('A', 'B')]
    tracks = bench.run_spectral_ab(prepared, tmp_path, ['off', 'audit', 'filter'], profile='fast', runner=runner)
    for track in ('A', 'B'):
        specs = [c for c in calls if c['mix_path'] == f'/m/{track}.wav']
        assert [c['mode'] for c in specs] == ['off', 'off', 'audit', 'filter']
        assert {c['cache_root'] for c in specs} == {str(tmp_path / track)}
        assert all(c['ground_truth_midi'] == f'/g/{track}.mid' and c['profile'] == 'fast' for c in specs)
    assert list(tracks[0]['runs']) == ['off_cold', 'off', 'audit', 'filter']
    assert tracks[0]['runs']['filter']['false_notes_on_absent_instrument'] == {'vocals': 4}


def test_summary_deltas_and_per_track_distribution():
    tracks = [_track('a', filter=_run(f1=.77, recall=.75, time=12.0, rss=1100.0)),
              _track('b', filter=_run(f1=.81, recall=.79, time=11.0, rss=1050.0)),
              _track('c', filter=_run(f1=.80, recall=.80))]
    summary = bench.summarize_spectral_ab(tracks, ['off', 'filter'])
    comp = summary['comparisons']['filter']
    assert comp['delta']['pitched_f1'] == pytest.approx((-.03 + .01 + 0) / 3)
    dist = comp['f1_distribution']
    assert (dist['min'], dist['median'], dist['max']) == pytest.approx((-.03, 0.0, .01))
    assert dist['recall_drops_over_0_02'] == 1
    assert summary['modes']['filter']['pitched_f1'] == pytest.approx((.77 + .81 + .80) / 3)
    acc = comp['acceptance']
    assert acc['extra_time']['value'] == pytest.approx(3.0 / 300.0)
    assert acc['extra_ram']['value'] == pytest.approx(100.0)
    assert acc['f1']['status'] == 'FAIL' and acc['extra_time']['status'] == 'PASS'


def test_recall_boundary_and_missing_data():
    def status(recall):
        tracks = [_track('a', filter=_run(recall=.8 - recall))]
        return bench.summarize_spectral_ab(tracks, ['off', 'filter'])['comparisons']['filter']['acceptance']
    assert status(.02)['recall']['status'] == 'PASS'
    assert status(.021)['recall']['status'] == 'FAIL'
    missing = bench.summarize_spectral_ab([_track('a', filter=_run(error='boom'))], ['off', 'filter'])
    acc = missing['comparisons']['filter']['acceptance']
    for name in ('f1', 'precision', 'recall', 'absent_false_notes', 'per_stem', 'extra_time', 'extra_ram', 'sdr'):
        assert acc[name]['status'] == 'NOT_MEASURED', name
    assert acc['errors']['status'] == 'FAIL'


def test_other_targets_boundaries():
    def acc(**kwargs):
        return bench.summarize_spectral_ab([_track('a', filter=_run(**kwargs))],
                                           ['off', 'filter'])['comparisons']['filter']['acceptance']
    fewer = {'bass': {'notes_count': 10}, 'vocals': {'notes_count': 8}}
    assert acc(stems=fewer)['absent_false_notes']['status'] == 'PASS'
    assert acc(stems={'bass': {'notes_count': 10}, 'vocals': {'notes_count': 9}})['absent_false_notes']['status'] == 'FAIL'
    assert acc(per_stem={'bass': .75})['per_stem']['status'] == 'PASS'
    assert acc(per_stem={'bass': .74})['per_stem']['status'] == 'FAIL'
    assert acc(time=25.0)['extra_time']['status'] == 'PASS'
    assert acc(time=25.1)['extra_time']['status'] == 'FAIL'
    assert acc(rss=1512.0)['extra_ram']['status'] == 'PASS'
    assert acc(rss=1513.0)['extra_ram']['status'] == 'FAIL'
    assert acc(sdr=10.005)['sdr']['status'] == 'PASS'
    assert acc(sdr=10.02)['sdr']['status'] == 'FAIL'
    no_absent = bench.summarize_spectral_ab(
        [_track('a', off=_run(stems={'bass': {'notes_count': 1}}), filter=_run(stems={'bass': {'notes_count': 1}}))],
        ['off', 'filter'])
    assert no_absent['comparisons']['filter']['acceptance']['absent_false_notes']['status'] == 'NOT_MEASURED'


def test_markdown_shows_acceptance_only_when_requested():
    tracks = [_track('a', filter=_run())]
    report = {'reference_tracks': [], 'manual_spotcheck': [], 'git_revision': 'x',
              'spectral_validation': {'modes_requested': ['off', 'filter'], 'split': 'test', 'tracks': tracks,
                                      **bench.summarize_spectral_ab(tracks, ['off', 'filter'])}}
    md = bench.render_markdown_report(report)
    assert '## Spectral validation A/B' in md and 'PASS' in md and 'recall' in md
    del report['spectral_validation']
    assert 'Spectral validation' not in bench.render_markdown_report(report)


def test_worker_entrypoint_runs_in_real_subprocess(tmp_path, monkeypatch):
    (tmp_path / 'bench_stub.py').write_text(
        'import json\nfrom pathlib import Path\n'
        'def run_pipeline(file_path, workspace=None, **kwargs):\n'
        '    (Path(workspace.cache_root) / "calls.json").write_text(json.dumps(kwargs))\n'
        'def evaluate_quality(workspace, file_path, **kwargs):\n'
        '    return {"transcription_metrics": {"pitched": {"onset_pitch_f1": 0.5}}}\n')
    monkeypatch.setenv('MUSIC_RE_BENCH_STUB', 'bench_stub')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    cache = tmp_path / 'cache'
    result = bench._run_worker({'mix_path': str(tmp_path / 'mix.wav'), 'cache_root': str(cache), 'mode': 'filter',
                                'profile': 'fast', 'six_stems': False, 'ground_truth_stems': None,
                                'ground_truth_midi': None})
    assert result['error'] is None, result
    assert result['quality_report']['transcription_metrics']['pitched']['onset_pitch_f1'] == .5
    assert result['wall_time_sec'] >= 0 and result['peak_rss_mb'] > 0
    assert json.loads((cache / 'calls.json').read_text())['spectral_validation'] == 'filter'


@pytest.mark.parametrize('extra', [[], ['--spectral-validation', 'off', 'filter', '--split', 'test']])
def test_main_wires_default_and_ab_paths(tmp_path, monkeypatch, extra):
    raw = tmp_path / 'raw'
    ids = [f'Track{i:05d}' for i in range(6)]
    for t in ids:
        (raw / t).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bench, 'prepare_track', lambda d, out: {
        'track_id': d.name, 'mix_path': str(d / 'mix.wav'), 'ground_truth_stems_dir': 'g',
        'ground_truth_midi_path': 'g.mid', 'group_status': {}, 'warnings': []})
    seen = {}
    monkeypatch.setattr(bench, 'run_reference_benchmarks', lambda tracks, **k: seen.setdefault(
        'ref', [dict(track_id=t['track_id']) for t in tracks]))
    monkeypatch.setattr(bench, 'run_spectral_ab', lambda tracks, root, modes, **k: seen.setdefault(
        'ab', [_track(t['track_id'], filter=_run()) | {'preparation_warnings': []} for t in tracks]))
    monkeypatch.setattr(sys, 'argv', ['bench', '--slakh-raw-dir', str(raw), *extra])
    bench.main()
    report = json.loads((tmp_path / 'BenchmarkReport.json').read_text())
    if not extra:
        assert 'spectral_validation' not in report and 'ab' not in seen
        assert [e['track_id'] for e in report['reference_tracks']] == ids
    else:
        assert 'ref' not in seen
        expected = [t for t in ids if bench.track_in_split(t, 'test')]
        assert [e['track_id'] for e in report['reference_tracks']] == expected
        assert report['spectral_validation']['comparisons']['filter']['acceptance']['recall']['status'] == 'PASS'
