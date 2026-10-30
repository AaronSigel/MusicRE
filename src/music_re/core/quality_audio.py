from pathlib import Path

import numpy as np

from music_re.core.audio_io import decode_audio

# Estimate stems that a 4-group reference (vocals/drums/bass/other) folds into 'other'.
OTHER_GROUP = ('other', 'guitar', 'piano')


def _channels(signal):
    return signal[:, None] if signal.ndim == 1 else signal


def _mono(signal):
    return _channels(signal).mean(axis=1, keepdims=True)


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _residual(metrics, mix, estimates):
    """Reconstruction residual on a common channel layout; excluded stems are listed."""
    mix = _channels(mix)
    excluded = sorted(metrics.pop('_decode_failed'))
    usable = dict(estimates)
    if any(v.shape[1] != mix.shape[1] for v in usable.values()):
        mix = _mono(mix)
        usable = {k: _mono(v) for k, v in usable.items()}
        metrics['residual_layout'] = 'mono_downmix'
    for name, value in list(usable.items()):
        if value.shape != mix.shape:
            excluded.append(name)
            del usable[name]
    metrics['residual_excluded_stems'] = excluded
    if not usable:
        return
    energy = float(np.sum(mix ** 2))
    residual = float(np.sum((mix - np.sum(list(usable.values()), axis=0)) ** 2))
    if energy > 0 and residual > 0:
        metrics['residual_reconstruction_db'] = float(10 * np.log10(residual / energy))
    elif energy > 0 and not excluded:
        metrics['warnings'].append('Perfect reconstruction: residual dB is negative infinity (stored as null).')
    elif energy > 0:
        metrics['warnings'].append(
            f'Zero residual over included stems only (excluded: {", ".join(excluded)}); stored as null.')


def _fold_other_group(estimates, directory, metrics):
    """Sum guitar/piano estimates into 'other' when the reference only has 'other'."""
    if not (directory / 'other.wav').is_file():
        return estimates
    folded = [n for n in OTHER_GROUP[1:] if n in estimates and not (directory / f'{n}.wav').is_file()]
    if not folded:
        return estimates
    members = [n for n in OTHER_GROUP if n in estimates and (n == 'other' or n in folded)]
    shapes = {estimates[n].shape for n in members}
    if len(shapes) != 1:
        metrics['warnings'].append(f'other: cannot sum {members}: shape mismatch')
        return estimates
    combined = {k: v for k, v in estimates.items() if k not in members}
    combined['other'] = np.sum([estimates[n] for n in members], axis=0)
    metrics['other_includes'] = members
    return combined


def separation_quality(mix, sr, stems, ground_truth=None):
    """BSSEval v4, fixed source order, one whole-track frame, no permutation.

    Per-stem ``stem_status``: scored, perfect (SDR=+inf, stored as null),
    undefined (nan / silent), missing_reference, failed.
    """
    metrics = dict(mode='diagnostic_reconstruction', comparisons=0, warnings=[],
                   stems_count=len(stems), residual_reconstruction_db=None,
                   residual_excluded_stems=[], _decode_failed=[])
    estimates, status = {}, {}
    for name, path in stems.items():
        try:
            signal, _ = decode_audio(path, target_sr=sr, normalize_db=None)
            estimates[name] = _channels(signal)
        except Exception as e:
            metrics['warnings'].append(f'{name}: {e}')
            metrics['_decode_failed'].append(name)
            status[name] = 'failed'
    _residual(metrics, mix, estimates)
    if not ground_truth:
        return metrics

    metrics['stem_status'] = status
    directory = Path(ground_truth)
    estimates = _fold_other_group(estimates, directory, metrics)
    references, compared, names = [], [], []
    for name, estimate in estimates.items():
        ref_path = directory / f'{name}.wav'
        if not ref_path.is_file():
            metrics['warnings'].append(f'{name}: reference missing')
            status[name] = 'missing_reference'
            continue
        try:
            reference, _ = decode_audio(ref_path, target_sr=sr, normalize_db=None)
            reference = _channels(reference)
            if reference.shape != estimate.shape or not reference.size:
                raise ValueError('reference/estimate duration or channel mismatch')
            if references and reference.shape != references[0].shape:
                raise ValueError('reference source shapes differ')
        except Exception as e:
            metrics['warnings'].append(f'{name}: {e}')
            status[name] = 'failed'
            continue
        if not np.any(reference) or not np.any(estimate):
            metrics['warnings'].append(f'{name}: silent reference or estimate: BSSEval unavailable')
            status[name] = 'undefined'
            continue
        references.append(reference)
        compared.append(estimate)
        names.append(name)
    if not names:
        metrics['warnings'].append('No comparable reference stems; objective separation unavailable.')
        return metrics
    import museval
    try:
        sdr, isr, sir, sar = museval.evaluate(np.array(references), np.array(compared),
                                             win=references[0].shape[0], hop=references[0].shape[0],
                                             mode='v4', padding=False)
    except Exception as e:
        metrics['warnings'].append(f'BSSEval unavailable: {e}')
        for name in names:
            status[name] = 'failed'
        return metrics
    raw_sdr = {name: float(np.nanmedian(sdr[i])) if np.any(~np.isnan(sdr[i])) else float('nan')
               for i, name in enumerate(names)}
    for name, value in raw_sdr.items():
        if np.isfinite(value):
            status[name] = 'scored'
        elif value == np.inf:
            status[name] = 'perfect'
        else:
            status[name] = 'undefined'
    scores = {key: {name: _finite(np.nanmedian(values[i])) if np.any(~np.isnan(values[i])) else None
                    for i, name in enumerate(names)}
              for key, values in [('sdr', sdr), ('isr', isr), ('sir', sir), ('sar', sar)]}
    finite_sdr = [v for v in scores['sdr'].values() if v is not None]
    metrics.update(scores)
    metrics['comparisons'] = sum(status[n] in ('scored', 'perfect') for n in names)
    metrics['average_sdr_db'] = float(np.mean(finite_sdr)) if finite_sdr else None
    metrics['average_sdr_count'] = len(finite_sdr)
    metrics['compared_stems'] = names
    metrics['settings'] = dict(library='museval', mode='v4', permutation=False,
                               sample_rate=sr, frame='whole_track')
    if metrics['comparisons']:
        metrics['mode'] = 'objective_ground_truth'
    if any(status[n] == 'perfect' for n in names):
        metrics['warnings'].append('Perfect SDR (+inf) is stored as null with stem_status "perfect".')
    if any(v is None and status.get(n) != 'perfect'
           for group in scores.values() for n, v in group.items()):
        metrics['warnings'].append('Undefined/nonfinite BSSEval scores are stored as null.')
    return metrics
