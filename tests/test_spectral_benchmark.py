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


def test_benchmark_uses_public_note_scorer():
    import pretty_midi
    from music_re.core import quality_midi
    ref = [pretty_midi.Note(velocity=100, pitch=60, start=0, end=.5), pretty_midi.Note(velocity=100, pitch=64, start=1, end=1.5)]
    est = [pretty_midi.Note(velocity=100, pitch=60, start=0, end=.5)]
    assert quality_midi.score_notes(ref, est) == {'precision': 1.0, 'recall': .5, 'f1': pytest.approx(2 / 3)}
    assert bench.score_notes is quality_midi.score_notes
    assert not hasattr(bench, '_score') and not hasattr(bench, 'compare_modes')


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
    assert bench.build_parser().parse_args(['--reuse-cache', '/tmp/cache']).reuse_cache == '/tmp/cache'
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


def test_reused_cache_disables_cold_extra_time_measurement():
    tracks = [_track('a', filter=_run(time=12.0))]
    summary = bench.summarize_spectral_ab(tracks, ['off', 'filter'], reused_cache=True)
    assert summary['comparisons']['filter']['acceptance']['extra_time']['status'] == 'NOT_MEASURED'


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


def test_main_reuses_requested_cache_and_records_it(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    (raw / 'Track00001').mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bench, 'prepare_track', lambda d, out: {
        'track_id': d.name, 'mix_path': 'mix.wav', 'ground_truth_stems_dir': 'g',
        'ground_truth_midi_path': 'g.mid', 'group_status': {}, 'warnings': []})
    roots = []
    monkeypatch.setattr(bench, 'run_spectral_ab', lambda tracks, root, modes, **k: roots.append(root) or [])
    cache = tmp_path / 'existing-cache'
    cache.mkdir()
    monkeypatch.setattr(sys, 'argv', ['bench', '--slakh-raw-dir', str(raw), '--reuse-cache', str(cache),
                                     '--spectral-validation', 'off', 'audit', '--out-json', 'out.json',
                                     '--out-md', 'out.md', '--history-dir', 'history'])
    bench.main()
    report = json.loads((tmp_path / 'out.json').read_text())
    assert roots == [cache]
    assert report['reused_cache'] == str(cache)
    assert str(cache) in (tmp_path / 'out.md').read_text()


def test_absent_reduction_needs_measured_counts_in_both_runs():
    def acc(off_stems, trial_stems):
        tracks = [_track('a', off=_run(stems=off_stems), filter=_run(stems=trial_stems))]
        return bench.summarize_spectral_ab(tracks, ['off', 'filter'])['comparisons']['filter']['acceptance']
    measured = {'bass': {'notes_count': 10}, 'vocals': {'notes_count': 10}}
    missing = {'bass': {'notes_count': 10}}
    assert acc(measured, missing)['absent_false_notes']['status'] == 'NOT_MEASURED'
    assert acc(missing, measured)['absent_false_notes']['status'] == 'NOT_MEASURED'
    assert acc(measured, {'bass': {'notes_count': 10}, 'vocals': {'notes_count': None}})[
        'absent_false_notes']['status'] == 'NOT_MEASURED'
    assert acc(measured, {'bass': {'notes_count': 10}, 'vocals': {'notes_count': 8}})[
        'absent_false_notes']['status'] == 'PASS'
    tracks = [_track('a', filter=_run(stems={'bass': {'notes_count': 10}, 'vocals': {'notes_count': 0}})),
              _track('b', filter=_run(stems=missing))]
    acc2 = bench.summarize_spectral_ab(tracks, ['off', 'filter'])['comparisons']['filter']['acceptance']
    assert acc2['absent_false_notes']['status'] == 'NOT_MEASURED'


def test_six_stem_worker_does_not_score_per_stem(tmp_path, monkeypatch):
    monkeypatch.delenv('MUSIC_RE_BENCH_STUB', raising=False)
    monkeypatch.setattr(bench, 'run_pipeline', lambda *a, **k: {})
    monkeypatch.setattr(bench, 'evaluate_quality', lambda *a, **k: {'transcription_metrics': {}})

    def never(*a, **k):
        raise AssertionError('six-stem output must not be scored against grouped references')
    monkeypatch.setattr(bench, '_per_stem_f1', never)
    result = bench.worker_main({'mix_path': 'm.wav', 'cache_root': str(tmp_path), 'mode': 'filter',
                                'six_stems': True, 'ground_truth_midi': 'g.mid'})
    assert result['error'] is None and result['per_stem_f1'] is None and 'six-stem' in result['per_stem_error']


def test_per_stem_not_measured_when_any_paired_run_unscored():
    unscored = dict(_run(), per_stem_f1=None)
    tracks = [_track('a', filter=_run(per_stem={'bass': .8})), _track('b', off=unscored, filter=unscored)]
    acc = bench.summarize_spectral_ab(tracks, ['off', 'filter'])['comparisons']['filter']['acceptance']
    assert acc['per_stem']['status'] == 'NOT_MEASURED'


@pytest.mark.parametrize('kind', ['missing', 'file'])
def test_main_rejects_reuse_cache_that_is_not_a_directory(tmp_path, monkeypatch, capsys, kind):
    raw = tmp_path / 'raw'
    (raw / 'Track00001').mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    prepared = []
    monkeypatch.setattr(bench, 'prepare_track', lambda d, out: prepared.append(d))
    cache = tmp_path / 'no-such-cache'
    if kind == 'file':
        cache.write_text('')
    monkeypatch.setattr(sys, 'argv', ['bench', '--slakh-raw-dir', str(raw), '--reuse-cache', str(cache),
                                     '--out-json', 'out.json', '--out-md', 'out.md', '--history-dir', 'history'])
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code != 0
    assert f'--reuse-cache {cache} is not an existing directory' in capsys.readouterr().err
    assert prepared == [] and not (tmp_path / 'out.json').exists()
