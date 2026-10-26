from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi


def _notes(pm, drums):
    return [n for inst in pm.instruments if inst.is_drum == drums for n in inst.notes]


def _arrays(notes):
    return (np.array([[n.start, n.end] for n in notes], dtype=float).reshape(-1, 2),
            np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes], dtype=float))


def _score(reference, estimate, offsets):
    intervals, pitches = _arrays(reference)
    est_intervals, est_pitches = _arrays(estimate)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        intervals, pitches, est_intervals, est_pitches, onset_tolerance=.05,
        pitch_tolerance=50.0, offset_ratio=.2 if offsets else None, offset_min_tolerance=.05)
    return dict(precision=float(p), recall=float(r), f1=float(f))


def transcription_quality(midis, ground_truth=None):
    metrics = dict(mode='diagnostic_note_density', comparisons=0, total_notes=0, stems={}, warnings=[])
    estimates = []
    seen = set()
    for role, path in midis.items():
        if role == 'all_stems':
            continue
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            pm = pretty_midi.PrettyMIDI(str(resolved))
            notes = [n for i in pm.instruments for n in i.notes]
            metrics['total_notes'] += len(notes)
            metrics['stems'][role] = dict(notes_count=len(notes),
                                          pitch_min=min((n.pitch for n in notes), default=None),
                                          pitch_max=max((n.pitch for n in notes), default=None))
            estimates.append(pm)
        except Exception as e:
            metrics['warnings'].append(f'{role}: invalid MIDI: {e}')
    if not ground_truth:
        return metrics
    if not estimates:
        metrics['warnings'].append('No valid estimated MIDI for reference comparison.')
        return metrics
    try:
        reference = pretty_midi.PrettyMIDI(str(ground_truth))
        for drums, key in [(False, 'pitched'), (True, 'drums')]:
            ref_notes = _notes(reference, drums)
            est_notes = [n for pm in estimates for n in _notes(pm, drums)]
            if not ref_notes and not est_notes:
                continue
            onset = _score(ref_notes, est_notes, offsets=False)
            group = {f'onset_pitch_{k}': v for k, v in onset.items()}
            if not drums:
                group.update({f'onset_pitch_offset_{k}': v
                              for k, v in _score(ref_notes, est_notes, offsets=True).items()})
            metrics[key] = group
            metrics['comparisons'] += 1
        if metrics['comparisons']:
            metrics['mode'] = 'objective_ground_truth'
        else:
            metrics['warnings'].append('Both reference and estimate contain no notes; score unavailable.')
        metrics['settings'] = dict(library='mir_eval.transcription', onset_tolerance_sec=.05,
                                   pitch_tolerance_cents=50, offset_ratio=.2,
                                   offset_min_tolerance_sec=.05, drums='GM pitch and onset, no offsets')
    except Exception as e:
        metrics['warnings'].append(f'Reference comparison unavailable: {e}')
    return metrics
