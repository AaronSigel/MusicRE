import json

import numpy as np
import soundfile as sf

from music_re.core.quality_audio import separation_quality

SR = 22050


def _signal(seed, channels=None, n=SR):
    rng = np.random.RandomState(seed)
    shape = (n,) if channels is None else (n, channels)
    return (0.1 * rng.randn(*shape)).astype(np.float32)


def _write(path, data):
    sf.write(path, data, SR, subtype='FLOAT')
    return str(path)


def test_mono_mix_with_stereo_stems_gives_finite_residual(tmp_path):
    a, b = _signal(1, 2), _signal(2, 2)
    mix = (a + b).mean(axis=1) + _signal(3) * 0.1
    stems = {'vocals': _write(tmp_path / 'v.wav', a), 'other': _write(tmp_path / 'o.wav', b)}
    metrics = separation_quality(mix, SR, stems)
    assert metrics['residual_reconstruction_db'] is not None
    assert np.isfinite(metrics['residual_reconstruction_db'])
    assert metrics['residual_excluded_stems'] == []
    json.dumps(metrics, allow_nan=False)


def test_stereo_mix_with_mono_stems_gives_finite_residual(tmp_path):
    a, b = _signal(1), _signal(2)
    mix = np.stack([a + b, a + b + _signal(3) * 0.1], axis=1)
    stems = {'vocals': _write(tmp_path / 'v.wav', a), 'other': _write(tmp_path / 'o.wav', b)}
    metrics = separation_quality(mix, SR, stems)
    assert metrics['residual_reconstruction_db'] is not None
    assert np.isfinite(metrics['residual_reconstruction_db'])


def test_excluded_stem_suppresses_perfect_reconstruction_warning(tmp_path):
    a = _signal(1)
    mix = a.copy()
    stems = {'vocals': _write(tmp_path / 'v.wav', a),
             'other': _write(tmp_path / 'o.wav', _signal(2, n=SR // 2))}
    metrics = separation_quality(mix, SR, stems)
    assert metrics['residual_excluded_stems'] == ['other']
    assert not any('Perfect reconstruction' in w for w in metrics['warnings'])
    assert metrics['residual_reconstruction_db'] is None


def test_perfect_reconstruction_warning_without_exclusions(tmp_path):
    a = _signal(1)
    stems = {'vocals': _write(tmp_path / 'v.wav', a)}
    metrics = separation_quality(a.copy(), SR, stems)
    assert metrics['residual_excluded_stems'] == []
    assert any('Perfect reconstruction' in w for w in metrics['warnings'])


def test_identical_estimate_is_perfect_and_counted(tmp_path):
    gt = tmp_path / 'gt'
    gt.mkdir()
    a = _signal(1)
    _write(gt / 'vocals.wav', a)
    stems = {'vocals': _write(tmp_path / 'v.wav', a)}
    metrics = separation_quality(a, SR, stems, gt)
    assert metrics['stem_status']['vocals'] == 'perfect'
    assert metrics['sdr']['vocals'] is None
    assert metrics['comparisons'] >= 1
    assert metrics['mode'] == 'objective_ground_truth'
    json.dumps(metrics, allow_nan=False)


def test_stem_statuses_scored_undefined_missing_failed(tmp_path):
    gt = tmp_path / 'gt'
    gt.mkdir()
    ref = _signal(1)
    _write(gt / 'vocals.wav', ref)
    _write(gt / 'drums.wav', np.zeros(SR, dtype=np.float32))
    stems = {
        'vocals': _write(tmp_path / 'v.wav', ref + 0.05 * _signal(2)),
        'drums': _write(tmp_path / 'd.wav', _signal(3)),
        'bass': _write(tmp_path / 'b.wav', _signal(4)),
        'other': str(tmp_path / 'does_not_exist.wav'),
    }
    metrics = separation_quality(ref, SR, stems, gt)
    status = metrics['stem_status']
    assert status['vocals'] == 'scored'
    assert status['drums'] == 'undefined'
    assert status['bass'] == 'missing_reference'
    assert status['other'] == 'failed'
    assert metrics['comparisons'] == 1
    assert metrics['average_sdr_count'] == 1
    assert np.isclose(metrics['average_sdr_db'], metrics['sdr']['vocals'])


def test_six_stem_estimates_sum_into_reference_other(tmp_path, monkeypatch):
    import museval

    captured = {}

    def fake_evaluate(references, estimates, **kwargs):
        captured['references'] = np.array(references)
        captured['estimates'] = np.array(estimates)
        n = len(references)
        values = np.full((n, 1), 5.0)
        return values, values, values, values

    monkeypatch.setattr(museval, 'evaluate', fake_evaluate)

    gt = tmp_path / 'gt'
    gt.mkdir()
    other_ref, vocals_ref = _signal(1), _signal(2)
    _write(gt / 'other.wav', other_ref)
    _write(gt / 'vocals.wav', vocals_ref)
    other_est, guitar_est, piano_est = _signal(3), _signal(4), _signal(5)
    stems = {
        'vocals': _write(tmp_path / 'v.wav', vocals_ref),
        'other': _write(tmp_path / 'o.wav', other_est),
        'guitar': _write(tmp_path / 'g.wav', guitar_est),
        'piano': _write(tmp_path / 'p.wav', piano_est),
    }
    metrics = separation_quality(other_ref + vocals_ref, SR, stems, gt)

    assert metrics['other_includes'] == ['other', 'guitar', 'piano']
    names = metrics['compared_stems']
    assert 'other' in names
    assert 'guitar' not in names and 'piano' not in names
    idx = names.index('other')
    expected = (other_est + guitar_est + piano_est)[:, None]
    assert np.allclose(captured['estimates'][idx], expected, atol=1e-6)
    assert np.allclose(captured['references'][idx], other_ref[:, None], atol=1e-6)
    assert metrics['stem_status']['other'] == 'scored'
    assert 'guitar' not in metrics['stem_status']
    assert not any('guitar: reference missing' in w for w in metrics['warnings'])
