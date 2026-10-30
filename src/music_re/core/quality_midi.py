from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

# Single source of truth for drum classes: canonical detector pitch and all valid GM aliases.
# Transcribers emit canonical pitches (e.g. 36/38/42/47/49),
# but ground-truth references use the full GM drum kit, where the same class has several
# valid pitches (e.g. kick 35 as well as 36, ride 51 as well as crash 49).
# Score each class against every GM pitch that belongs to it.
DRUM_CLASS_DEFINITIONS = {
    'kick': {'canonical': 36, 'gm_pitches': {35, 36}},
    'snare': {'canonical': 38, 'gm_pitches': {37, 38, 40}},
    'hihat': {'canonical': 42, 'gm_pitches': {42, 44, 46}},
    'cymbal': {'canonical': 49, 'gm_pitches': {49, 51, 52, 53, 55, 57, 59}},
    'tom': {'canonical': 47, 'gm_pitches': {41, 43, 45, 47, 48, 50}},
}
DRUM_GM_PITCH_CLASSES = {k: v['gm_pitches'] for k, v in DRUM_CLASS_DEFINITIONS.items()}
CANONICAL_DRUM_PITCHES = {k: v['canonical'] for k, v in DRUM_CLASS_DEFINITIONS.items()}
assert DRUM_GM_PITCH_CLASSES.keys() == CANONICAL_DRUM_PITCHES.keys()
for _cls, _canon in CANONICAL_DRUM_PITCHES.items():
    assert _canon in DRUM_GM_PITCH_CLASSES[_cls]


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
            if drums:
                by_class = {}
                for name, pitches in DRUM_GM_PITCH_CLASSES.items():
                    canon_pitch = CANONICAL_DRUM_PITCHES[name]
                    ref_class = [
                        pretty_midi.Note(velocity=n.velocity, pitch=canon_pitch, start=n.start, end=n.end)
                        for n in ref_notes if n.pitch in pitches
                    ]
                    est_class = [
                        pretty_midi.Note(velocity=n.velocity, pitch=canon_pitch, start=n.start, end=n.end)
                        for n in est_notes if n.pitch in pitches
                    ]
                    if not ref_class and not est_class:
                        continue
                    by_class[name] = {**_score(ref_class, est_class, offsets=False),
                                      'ref_count': len(ref_class), 'est_count': len(est_class)}
                if by_class:
                    metrics['drums_by_class'] = by_class
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
