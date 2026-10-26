from pathlib import Path

import numpy as np

from music_re.core.audio_io import decode_audio


def _channels(signal):
    return signal[:, None] if signal.ndim == 1 else signal


def _finite(value):
    return float(value) if np.isfinite(value) else None


def separation_quality(mix, sr, stems, ground_truth=None):
    """BSSEval v4, fixed source order, one whole-track frame, no permutation."""
    metrics = dict(mode='diagnostic_reconstruction', comparisons=0, warnings=[],
                   stems_count=len(stems), residual_reconstruction_db=None)
    estimates = {}
    for name, path in stems.items():
        try:
            signal, _ = decode_audio(path, target_sr=sr, normalize_db=None)
            estimates[name] = _channels(signal)
        except Exception as e:
            metrics['warnings'].append(f'{name}: {e}')
    matching = [v for v in estimates.values() if v.shape == _channels(mix).shape]
    if matching:
        energy = float(np.sum(mix ** 2))
        residual = float(np.sum((_channels(mix) - np.sum(matching, axis=0)) ** 2))
        if energy > 0 and residual > 0:
            metrics['residual_reconstruction_db'] = float(10 * np.log10(residual / energy))
        elif energy > 0:
            metrics['warnings'].append('Perfect reconstruction: residual dB is negative infinity (stored as null).')
    if not ground_truth:
        return metrics

    directory = Path(ground_truth)
    references, compared, names = [], [], []
    for name, estimate in estimates.items():
        ref_path = directory / f'{name}.wav'
        if not ref_path.is_file():
            metrics['warnings'].append(f'{name}: reference missing')
            continue
        try:
            reference, _ = decode_audio(ref_path, target_sr=sr, normalize_db=None)
            reference = _channels(reference)
            if reference.shape != estimate.shape or not reference.size:
                raise ValueError('reference/estimate duration or channel mismatch')
            if not np.any(reference) or not np.any(estimate):
                raise ValueError('silent reference or estimate: BSSEval unavailable')
            if references and reference.shape != references[0].shape:
                raise ValueError('reference source shapes differ')
            references.append(reference)
            compared.append(estimate)
            names.append(name)
        except Exception as e:
            metrics['warnings'].append(f'{name}: {e}')
    if not names:
        metrics['warnings'].append('No comparable reference stems; objective separation unavailable.')
        return metrics
    import museval
    try:
        sdr, isr, sir, sar = museval.evaluate(np.array(references), np.array(compared),
                                             win=references[0].shape[0], hop=references[0].shape[0],
                                             mode='v4', padding=False)
        scores = {key: {name: _finite(np.nanmedian(values[i])) for i, name in enumerate(names)}
                  for key, values in [('sdr', sdr), ('isr', isr), ('sir', sir), ('sar', sar)]}
        finite_sdr = [v for v in scores['sdr'].values() if v is not None]
        metrics.update(scores)
        metrics['comparisons'] = len(finite_sdr)
        metrics['average_sdr_db'] = float(np.mean(finite_sdr)) if finite_sdr else None
        metrics['compared_stems'] = names
        metrics['settings'] = dict(library='museval', mode='v4', permutation=False,
                                   sample_rate=sr, frame='whole_track')
        if finite_sdr:
            metrics['mode'] = 'objective_ground_truth'
        if any(v is None for group in scores.values() for v in group.values()):
            metrics['warnings'].append('Undefined/nonfinite BSSEval scores are stored as null.')
    except Exception as e:
        metrics['warnings'].append(f'BSSEval unavailable: {e}')
    return metrics
