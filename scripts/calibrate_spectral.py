"""Calibrate the per-role spectral note score from benchmark caches.

Reads filter-mode spectral_validation.json reports (per-note evidence) from a
quality-benchmark cache root, labels each transcribed note against the
octave-corrected Slakh reference (TP iff a reference note of the same group
with equal pitch starts within 50 ms), fits a standardised logistic
regression per role on the tune split only, and picks the highest reject
threshold whose tune TP loss stays within the recall cap. Tune and test
metrics are reported side by side, together with the current threshold rule.

The ``presence`` subcommand calibrates the stem presence rule (absent-instrument
detection). Present examples: MUSDB18 7-s previews whose reference vocals are
louder than -40 dBFS, run through the pipeline (audit mode); BabySlakh stems
whose group is present. Absent examples: BabySlakh stems whose group is absent
(vocals on every track) and MUSDB previews with silent reference vocals. The
thresholds keep every tune present example present and maximise tune absent
recall; test results are reported separately.

Usage:
    uv run --locked python scripts/calibrate_spectral.py \
        --cache-root data/benchmarks/cache/<run> --raw-dir data/benchmarks/slakh_raw/raw \
        --prepared-dir data/benchmarks/calibration/prepared \
        --out data/benchmarks/calibration/spectral_calibration.json
    uv run --locked python scripts/calibrate_spectral.py presence \
        --slakh-cache data/benchmarks/cache/<run> --prepared-dir data/benchmarks/calibration/prepared \
        --musdb-dir data/benchmarks/musdb7 --musdb-cache data/benchmarks/calibration/musdb_cache \
        --out data/benchmarks/calibration/presence_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / 'src'))

from music_re.core.midi_validation import (PRESENCE_ROLES, SCORE_FEATURES,  # noqa: E402
                                           SpectralMidiValidator, score_features)
from music_re.schemas import NoteEvidence  # noqa: E402
from scripts.run_quality_benchmarks import track_in_split  # noqa: E402
from scripts.slakh_mapping import prepare_track  # noqa: E402

ONSET_TOLERANCE_SEC = 0.05
ROLE_GROUP = {'bass': 'bass', 'other': 'other', 'guitar': 'other', 'piano': 'other'}
NO_EVIDENCE_REASONS = ('out_of_range', 'invalid_duration')
SCENARIOS = ('unfiltered', 'current_rule', 'model')


@dataclass
class Row:
    track: str
    role: str
    split: str
    label: int
    features: list
    fit: bool  # False for notes without measurable evidence (never removed by score)
    decision: str
    matches: tuple  # indices of the reference notes this note matches


def is_tp(ref_by_pitch: dict, pitch: int, start: float) -> bool:
    starts = ref_by_pitch.get(pitch)
    return starts is not None and len(starts) > 0 and float(np.min(np.abs(starts - start))) <= ONSET_TOLERANCE_SEC


def _reference(track: str, prepared_dir: Path, raw_dir: Path | None) -> pretty_midi.PrettyMIDI:
    path = prepared_dir / track / 'ground_truth.mid'
    if not path.is_file():
        if raw_dir is None:
            raise FileNotFoundError(f'{path} missing and no --raw-dir to prepare it from')
        result = prepare_track(Path(raw_dir) / track, prepared_dir / track)
        (prepared_dir / track / 'prepare_result.json').write_text(json.dumps(result, indent=2, default=str))
    return pretty_midi.PrettyMIDI(str(path))


def collect(cache_root, prepared_dir, raw_dir, roles):
    """Rows for every scored note plus reference note counts per (track, role)."""
    cache_root, prepared_dir = Path(cache_root), Path(prepared_dir)
    rows, ref_counts = [], {}
    for report_path in sorted(cache_root.glob('*/*/spectral_validation.json')):
        track = report_path.parent.parent.name
        split = 'tune' if track_in_split(track, 'tune') else 'test'
        report = json.loads(report_path.read_text())
        ref = _reference(track, prepared_dir, raw_dir)
        for role in roles:
            stem = report['stems'].get(role)
            if stem is None or stem.get('status') != 'validated':
                continue
            group = ROLE_GROUP[role]
            ref_notes = [n for inst in ref.instruments if inst.name == group for n in inst.notes]
            ref_counts[(track, role)] = ref_counts.get((track, role), 0) + len(ref_notes)
            by_pitch = defaultdict(list)
            for i, n in enumerate(ref_notes):
                by_pitch[n.pitch].append((n.start, i))
            starts = {p: np.array([s for s, _ in v]) for p, v in by_pitch.items()}
            for note in stem['notes']:
                item = NoteEvidence(**note)
                matches = tuple(i for s, i in by_pitch.get(item.pitch, ())
                                if abs(s - item.start_sec) <= ONSET_TOLERANCE_SEC)
                rows.append(Row(track=track, role=role, split=split,
                                label=int(is_tp(starts, item.pitch, item.start_sec)),
                                features=score_features(item).tolist(),
                                fit=item.reason not in NO_EVIDENCE_REASONS,
                                decision=item.decision, matches=matches))
    return rows, ref_counts


def collect_rows(cache_root, prepared_dir, raw_dir, roles):
    return collect(cache_root, prepared_dir, raw_dir, roles)[0]


def score(params: dict, x: np.ndarray) -> np.ndarray:
    """Same formula as music_re.core.midi_validation.note_score."""
    z = (np.asarray(x, dtype=np.float64) - np.asarray(params['mean'])) / np.asarray(params['scale'])
    logit = np.clip(z @ np.asarray(params['weights']) + params['bias'], -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-logit))


def pick_threshold(scores: np.ndarray, labels: np.ndarray, cap: float) -> float:
    """Highest t such that removing notes with score < t loses at most `cap` of the TPs."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    total = labels.sum()
    best = float(np.min(scores))
    for t in np.unique(scores):
        lost = labels[scores < t].sum() / total if total else 0.0
        if lost <= cap + 1e-12:
            best = float(t)
        else:
            break
    return best


def rule_params(role_result: dict) -> dict:
    """RoleRule keyword arguments for a calibrated role."""
    return dict(score_weights=tuple(role_result['weights']), score_bias=role_result['bias'],
                score_mean=tuple(role_result['mean']), score_scale=tuple(role_result['scale']),
                reject_below=role_result['reject_below'], uncertain_below=role_result['uncertain_below'])


def _prf(tp_pred, n_pred, matched, n_ref):
    p = tp_pred / n_pred if n_pred else 0.0
    r = matched / n_ref if n_ref else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)


def _keep_masks(rows, params):
    x = np.array([r.features for r in rows]) if rows else np.zeros((0, len(SCORE_FEATURES)))
    fit = np.array([r.fit for r in rows], dtype=bool)
    scores = score(params, x) if len(rows) else np.zeros(0)
    return {'unfiltered': np.ones(len(rows), dtype=bool),
            'current_rule': np.array([r.decision != 'reject' for r in rows], dtype=bool),
            'model': ~fit | (scores >= params['reject_below'])}


def _relative(rows, params):
    fit_rows = [r for r in rows if r.fit]
    y = np.array([r.label for r in fit_rows])
    masks = _keep_masks(fit_rows, params)
    out = {'n': len(fit_rows), 'n_tp': int(y.sum())}
    for name, keep in masks.items():
        precision = float(y[keep].mean()) if keep.any() else 0.0
        kept = float(y[keep].sum() / y.sum()) if y.sum() else 0.0
        out[name] = dict(precision=precision, tp_kept=kept, removed=float(1 - keep.mean()) if len(y) else 0.0,
                         f1_proxy=2 * precision * kept / (precision + kept) if precision + kept else 0.0)
    try:
        from sklearn.metrics import roc_auc_score
        x = np.array([r.features for r in fit_rows])
        out['auc'] = float(roc_auc_score(y, score(params, x))) if 0 < y.sum() < len(y) else None
    except ValueError:
        out['auc'] = None
    return out


def _absolute(rows, ref_counts, params):
    """Pitched note P/R/F1 against the full reference, per track and summed."""
    per_track = {}
    by_track = defaultdict(list)
    for r in rows:
        by_track[r.track].append(r)
    for track, track_rows in by_track.items():
        masks = _keep_masks(track_rows, params)
        n_ref = ref_counts.get((track, track_rows[0].role), 0)
        per_track[track] = {}
        for name, keep in masks.items():
            kept = [r for r, k in zip(track_rows, keep) if k]
            matched = len({i for r in kept for i in r.matches})
            per_track[track][name] = dict(tp_pred=sum(r.label for r in kept), n_pred=len(kept),
                                          matched=matched, n_ref=n_ref)
    totals = {}
    for name in SCENARIOS:
        sums = defaultdict(int)
        for counts in per_track.values():
            for k, v in counts[name].items():
                sums[k] += v
        totals[name] = _prf(sums['tp_pred'], sums['n_pred'], sums['matched'], sums['n_ref'])
    per_track = {t: {name: _prf(**c[name]) for name in SCENARIOS} for t, c in per_track.items()}
    return totals, per_track


def fit_role(rows, cap):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tune = [r for r in rows if r.split == 'tune' and r.fit]
    y = np.array([r.label for r in tune])
    if len(tune) == 0 or y.min() == y.max():
        return None
    x = np.array([r.features for r in tune])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(x), y)
    scale = [float(s) if s > 0 else 1.0 for s in scaler.scale_]
    params = dict(weights=[float(w) for w in model.coef_[0]], bias=float(model.intercept_[0]),
                  mean=[float(m) for m in scaler.mean_], scale=scale)
    reject = pick_threshold(score(params, x), y, cap)
    params['reject_below'] = reject
    params['uncertain_below'] = (reject + 0.5) / 2 if reject < 0.5 else reject
    return params


def _recommend(test: dict, cap: float):
    model, base, current = test['model'], test['unfiltered'], test['current_rule']
    if test['n'] == 0:
        return False, 'no test notes'
    if model['tp_kept'] < 1 - cap - 1e-12:
        return False, f"test TP loss {1 - model['tp_kept']:.4f} exceeds cap {cap}"
    if model['precision'] <= base['precision']:
        return False, f"no test precision gain ({base['precision']:.4f} -> {model['precision']:.4f})"
    if model['f1_proxy'] < current['f1_proxy']:
        return False, (f"test F1 proxy {model['f1_proxy']:.4f} below current rule "
                       f"{current['f1_proxy']:.4f}")
    return True, (f"test precision {base['precision']:.4f} -> {model['precision']:.4f}, "
                  f"TP kept {model['tp_kept']:.4f}")


def calibrate(cache_root, prepared_dir, raw_dir, roles=('bass', 'other'), recall_cap=0.02) -> dict:
    rows, ref_counts = collect(cache_root, prepared_dir, raw_dir, roles)
    result = {'features': list(SCORE_FEATURES), 'recall_cap': recall_cap,
              'onset_tolerance_sec': ONSET_TOLERANCE_SEC, 'roles': {}, 'per_track': []}
    for role in roles:
        role_rows = [r for r in rows if r.role == role]
        params = fit_role(role_rows, recall_cap)
        if params is None:
            result['roles'][role] = {'recommend_update': False,
                                     'recommend_reason': 'tune split lacks both classes'}
            continue
        metrics = {}
        for split in ('tune', 'test'):
            split_rows = [r for r in role_rows if r.split == split]
            metrics[split] = _relative(split_rows, params)
            metrics[split]['absolute'], per_track = _absolute(split_rows, ref_counts, params)
            for track, values in sorted(per_track.items()):
                result['per_track'].append({'track': track, 'role': role, 'split': split, **values})
        ok, reason = _recommend(metrics['test'], recall_cap)
        result['roles'][role] = {**params, 'metrics': metrics, 'recommend_update': ok,
                                 'recommend_reason': reason}
    return result


def format_table(result: dict) -> str:
    lines = ['| role | split | n | TP rate | AUC | scenario | precision | TP kept | removed | '
             'abs P | abs R | abs F1 |', '|' + '---|' * 12]
    for role, data in result['roles'].items():
        for split, m in data.get('metrics', {}).items():
            for name in SCENARIOS:
                s, a = m[name], m['absolute'][name]
                auc = f"{m['auc']:.3f}" if m['auc'] is not None else '-'
                lines.append(f"| {role} | {split} | {m['n']} | {m['n_tp'] / max(m['n'], 1):.3f} | {auc} | "
                             f"{name} | {s['precision']:.4f} | {s['tp_kept']:.4f} | {s['removed']:.4f} | "
                             f"{a['precision']:.4f} | {a['recall']:.4f} | {a['f1']:.4f} |")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Stem presence (absent-instrument) calibration
# ---------------------------------------------------------------------------

MUSDB_PRESENT_DBFS = -40.0
MUSDB_ABSENT_DBFS = -60.0
NEVER_ABSENT = {'min_rel_db': -200.0, 'min_median_support': 0.0}


@dataclass
class PresenceExample:
    source: str  # 'musdb' or 'slakh'
    item: str    # clip or track id
    role: str
    split: str
    label: str   # 'present' or 'absent'
    rel_db: float | None
    median_support: float
    active_fraction: float
    n_notes: int


def musdb_label(vocals_db: float):
    """'present' above -40 dBFS, 'absent' below -60 dBFS, None (excluded) in between."""
    if vocals_db > MUSDB_PRESENT_DBFS:
        return 'present'
    if vocals_db < MUSDB_ABSENT_DBFS:
        return 'absent'
    return None


def is_absent(example: PresenceExample, rule: dict) -> bool:
    return (example.rel_db is not None and example.rel_db < rule['min_rel_db']
            and example.median_support < rule['min_median_support'])


def _midpoints(values):
    v = np.unique(np.asarray(values, dtype=np.float64))
    return list((v[:-1] + v[1:]) / 2) + [float(v[-1]) + 1e-6] if v.size else []


def choose_presence_thresholds(examples) -> dict:
    """Thresholds with zero false-absent on tune present examples and maximal tune absent recall.

    Candidates are midpoints between observed tune values. Ties go to the most
    conservative pair (smallest min_rel_db, then smallest min_median_support).
    Without tune absent examples the rule never fires.
    """
    tune = [e for e in examples if e.split == 'tune' and e.rel_db is not None]
    absent = [e for e in tune if e.label == 'absent']
    present = [e for e in tune if e.label == 'present']
    if not absent:
        return dict(NEVER_ABSENT)
    best, best_hits = dict(NEVER_ABSENT), 0
    for a in _midpoints([e.rel_db for e in tune]):
        for b in _midpoints([e.median_support for e in tune]):
            rule = {'min_rel_db': float(a), 'min_median_support': float(b)}
            if any(is_absent(e, rule) for e in present):
                continue
            hits = sum(is_absent(e, rule) for e in absent)
            if hits > best_hits:
                best, best_hits = rule, hits
    return best


def presence_metrics(examples, rule) -> dict:
    out = {}
    for split in ('tune', 'test'):
        rows = [e for e in examples if e.split == split]
        present = [e for e in rows if e.label == 'present']
        absent = [e for e in rows if e.label == 'absent']
        detected = sum(is_absent(e, rule) for e in absent)
        out[split] = dict(n_present=len(present), n_absent=len(absent),
                          false_absent=sum(is_absent(e, rule) for e in present),
                          absent_detected=detected,
                          absent_recall=detected / len(absent) if absent else None)
    return out


def _example(source, item, role, split, label, presence, n_notes):
    return PresenceExample(source=source, item=item, role=role, split=split, label=label,
                           rel_db=presence['rel_db'], median_support=presence['median_support'],
                           active_fraction=presence['active_fraction'], n_notes=n_notes)


def slakh_presence_examples(cache_root, prepared_dir, roles=('vocals', 'bass', 'other')):
    """Audit-mode presence features of cached BabySlakh stems (cache dirs are only read)."""
    validator = SpectralMidiValidator()
    examples = []
    for manifest_path in sorted(Path(cache_root).glob('*/*/manifest.json')):
        track = manifest_path.parent.parent.name
        manifest = json.loads(manifest_path.read_text())
        status = json.loads((Path(prepared_dir) / track / 'prepare_result.json').read_text())['group_status']
        split = 'tune' if track_in_split(track, 'tune') else 'test'
        for role in roles:
            label = {'present': 'present', 'absent': 'absent'}.get(status.get(role))
            if label is None or role not in manifest['midis']:
                continue
            report = validator.validate(manifest['stems'][role], manifest['midis'][role], role, 'audit',
                                        mix_audio_path=manifest['input_file'])
            examples.append(_example('slakh', track, role, split, label, report.presence, len(report.notes)))
    return examples


def _vocals_db(mixture_path, vocals_path) -> float:
    from music_re.core.midi_validation import _active_db, _frame_db
    from music_re.core.spectral import SpectralConfig, load_mono

    config = SpectralConfig()
    mix_db = _frame_db(load_mono(mixture_path, config.sr), config)
    vocals = _frame_db(load_mono(vocals_path, config.sr), config)
    n = min(len(mix_db), len(vocals))
    active = mix_db[:n] > -60.0
    return float(_active_db(vocals[:n][active])) if active.any() else -120.0


def musdb_presence_examples(musdb_dir, cache_dir, profile='fast'):
    """Run the pipeline (audit) on each labelled MUSDB preview; vocals presence from its report."""
    from music_re.core.pipeline import run_pipeline
    from music_re.workspace import WorkspaceManager

    examples, levels = [], {}
    for clip in json.loads((Path(musdb_dir) / 'subset.json').read_text()):
        level = _vocals_db(clip['mixture_path'], clip['vocals_path'])
        label = musdb_label(level)
        levels[clip['clip_id']] = dict(vocals_db=level, label=label)
        if label is None:
            continue
        ws = WorkspaceManager(cache_root=Path(cache_dir) / clip['clip_id'])
        run_pipeline(clip['mixture_path'], workspace=ws, profile=profile, spectral_validation='audit')
        report = json.loads(ws.get_spectral_validation_path(Path(clip['mixture_path']).resolve()).read_text())
        stem = report['stems']['vocals']
        examples.append(_example('musdb', clip['clip_id'], 'vocals', clip['split'], label,
                                 stem['presence'], len(stem['notes'])))
    return examples, levels


def calibrate_presence(examples) -> dict:
    """Per role thresholds; roles without own absent examples inherit the strictest calibrated rule."""
    result = {'roles': {}}
    for role in sorted({e.role for e in examples}):
        rows = [e for e in examples if e.role == role]
        rule = choose_presence_thresholds(rows)
        result['roles'][role] = dict(rule=rule, metrics=presence_metrics(rows, rule))
    calibrated = [r['rule'] for r in result['roles'].values() if r['rule'] != NEVER_ABSENT]
    if calibrated:
        strict = {'min_rel_db': min(r['min_rel_db'] for r in calibrated),
                  'min_median_support': min(r['min_median_support'] for r in calibrated)}
        for role in PRESENCE_ROLES:
            entry = result['roles'].get(role)
            if entry is None or entry['rule'] == NEVER_ABSENT:
                rows = [e for e in examples if e.role == role]
                result['roles'][role] = dict(rule=strict, inherited=True, metrics=presence_metrics(rows, strict))
    result['examples'] = [e.__dict__ for e in examples]
    return result


def format_presence_table(result: dict) -> str:
    lines = ['| role | min_rel_db | min_median_support | split | present | absent | false-absent | '
             'absent detected | absent recall |', '|' + '---|' * 9]
    for role, data in sorted(result['roles'].items()):
        rule = data['rule']
        for split, m in data['metrics'].items():
            recall = f"{m['absent_recall']:.3f}" if m['absent_recall'] is not None else '-'
            lines.append(f"| {role}{' (inherited)' if data.get('inherited') else ''} | "
                         f"{rule['min_rel_db']:.2f} | {rule['min_median_support']:.4f} | {split} | "
                         f"{m['n_present']} | {m['n_absent']} | {m['false_absent']} | "
                         f"{m['absent_detected']} | {recall} |")
    return '\n'.join(lines)


def presence_main(argv) -> int:
    parser = argparse.ArgumentParser(description='Calibrate the stem presence rule')
    parser.add_argument('--slakh-cache', required=True, type=Path)
    parser.add_argument('--prepared-dir', required=True, type=Path)
    parser.add_argument('--musdb-dir', type=Path, default=None)
    parser.add_argument('--musdb-cache', type=Path, default=Path('data/benchmarks/calibration/musdb_cache'))
    parser.add_argument('--profile', default='fast')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    examples = slakh_presence_examples(args.slakh_cache, args.prepared_dir)
    levels = {}
    if args.musdb_dir is not None:
        musdb, levels = musdb_presence_examples(args.musdb_dir, args.musdb_cache, args.profile)
        examples += musdb
    result = calibrate_presence(examples)
    result['musdb_levels'] = levels
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(format_presence_table(result))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['presence']:
        return presence_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cache-root', required=True, type=Path)
    parser.add_argument('--raw-dir', type=Path, default=None)
    parser.add_argument('--prepared-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--roles', nargs='+', default=['bass', 'other'], choices=sorted(ROLE_GROUP))
    parser.add_argument('--recall-cap', type=float, default=0.02)
    args = parser.parse_args(argv)
    result = calibrate(args.cache_root, args.prepared_dir, args.raw_dir, tuple(args.roles), args.recall_cap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(format_table(result))
    for role, data in result['roles'].items():
        print(f"{role}: recommend_update={data['recommend_update']} ({data['recommend_reason']})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
