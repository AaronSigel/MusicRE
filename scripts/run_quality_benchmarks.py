#!/usr/bin/env python3
"""Orchestrate the Slakh benchmark baseline: run MusicRE's real pipeline
against a prepared Slakh subset (objective, mir_eval/museval-scored) and,
optionally, a few real tracks with no ground truth (manual spot check).
Writes BenchmarkReport.json / BenchmarkReport.md and a timestamped history
snapshot so later versions can be compared against this one.
"""
import argparse
import contextlib
import hashlib
import importlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from music_re.core.pipeline import run_pipeline
from music_re.core.quality import evaluate_quality
from music_re.core.quality_midi import score_notes
from music_re.workspace import WorkspaceManager, active_midis
from scripts.fetch_slakh_subset import fetch_babyslakh
from scripts.slakh_mapping import ABSENT, prepare_track


def track_in_split(track_id: str, split: str) -> bool:
    if split not in ('all', 'tune', 'test'):
        raise ValueError(f'Invalid split: {split}')
    return split == 'all' or (int(hashlib.sha1(track_id.encode()).hexdigest(), 16) % 2 == (split == 'test'))


def _fresh_cache_root(base: Path) -> Path:
    """A new, never-before-used cache directory under base.

    A fixed cache_root would let a later run resume cached stems/MIDI from
    an earlier run that used different code, making the history snapshot
    this run writes incomparable to the previous one (it would silently
    show stale outputs under a new git revision).
    """
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return Path(base) / token


def _as_dict(report) -> dict:
    return report.model_dump() if hasattr(report, "model_dump") else dict(report)


def _peak_rss_mb() -> float:
    """Process-wide peak resident set size, not just the Python/numpy heap.

    tracemalloc only sees allocations made through Python's own allocator;
    torch and ONNX Runtime allocate most of their memory outside it, so
    tracemalloc under-reports real peak RAM by a large factor during a
    pipeline run. ru_maxrss is KB on Linux, bytes on macOS.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _run_one(mix_path, workspace, ground_truth_stems=None, ground_truth_midi=None, profile="fast", six_stems=False,
             spectral_validation="off"):
    """Time one pipeline+evaluate run; never raises, reports error instead."""
    t0 = time.perf_counter()
    quality_report, error = None, None
    try:
        run_pipeline(mix_path, workspace=workspace, profile=profile, six_stems=six_stems,
                     spectral_validation=spectral_validation)
        report = evaluate_quality(workspace, mix_path, ground_truth_stems=ground_truth_stems,
                                   ground_truth_midi=ground_truth_midi)
        quality_report = _as_dict(report)
    except Exception as e:
        error = str(e)
    wall_time_sec = time.perf_counter() - t0
    return quality_report, error, wall_time_sec, _peak_rss_mb()


def _false_notes_on_absent_instrument(quality_report: Optional[dict], group_status: Dict[str, str]) -> Dict[str, int]:
    """Notes transcribed for groups the reference marks "absent".

    Only "absent" counts: a group whose reference is missing_annotation
    (audio, no MIDI) or not_rendered really is in the song, so notes found
    there are not false positives. A track with no group_status at all has
    unknown completeness and contributes nothing.
    """
    if quality_report is None:
        return {}
    stems = (quality_report.get("transcription_metrics") or {}).get("stems", {})
    return {group: stems[group]["notes_count"]
            for group, status in group_status.items()
            if status == ABSENT and group in stems}


def prepare_all_tracks(track_dirs: List[Path], prepared_dir: Path):
    """Run prepare_track() per track, isolating one bad track's failure
    (corrupt metadata.yaml, a missing stem wav, an alignment mismatch) so
    it doesn't abort every other track's benchmark run.

    Returns (prepared, failures): prepared is prepare_track()'s normal
    output list; failures is a list already shaped like a
    run_reference_benchmarks() entry (quality_report=None, the raised
    error's message) so it can be appended straight into the report.
    """
    prepared, failures = [], []
    for track_dir in track_dirs:
        try:
            prepared.append(prepare_track(track_dir, prepared_dir / track_dir.name))
        except Exception as e:
            failures.append(dict(track_id=track_dir.name, quality_report=None, wall_time_sec=0.0,
                                  peak_ram_mb=0.0, error=str(e), false_notes_on_absent_instrument={},
                                  group_status={}, preparation_warnings=[]))
    return prepared, failures


def run_reference_benchmarks(prepared_tracks: List[dict], cache_root: Path, profile: str = "fast", six_stems: bool = False) -> List[dict]:
    results = []
    for track in prepared_tracks:
        track_id = track["track_id"]
        ws = WorkspaceManager(cache_root=Path(cache_root) / track_id)
        quality_report, error, wall_time_sec, peak_ram_mb = _run_one(
            track["mix_path"], ws,
            ground_truth_stems=track["ground_truth_stems_dir"],
            ground_truth_midi=track["ground_truth_midi_path"],
            profile=profile,
            six_stems=six_stems,
        )
        results.append(dict(
            track_id=track_id, quality_report=quality_report, wall_time_sec=wall_time_sec,
            peak_ram_mb=peak_ram_mb, error=error,
            false_notes_on_absent_instrument=_false_notes_on_absent_instrument(
                quality_report, track.get("group_status", {})),
            group_status=dict(track.get("group_status", {})),
            preparation_warnings=list(track.get("warnings", [])),
        ))
    return results


def run_manual_spotcheck(real_tracks: List[Path], cache_root: Path, profile: str = "fast", six_stems: bool = False) -> List[dict]:
    results = []
    for path in real_tracks:
        path = Path(path)
        track_id = path.stem
        ws = WorkspaceManager(cache_root=Path(cache_root) / track_id)
        quality_report, error, wall_time_sec, peak_ram_mb = _run_one(str(path), ws, profile=profile, six_stems=six_stems)
        results.append(dict(track_id=track_id, quality_report=quality_report, wall_time_sec=wall_time_sec,
                             peak_ram_mb=peak_ram_mb, error=error))
    return results


WORKER_RESULT_PREFIX = "MUSIC_RE_BENCH_RESULT "
RECALL_DROP_LIMIT = 0.02
ABSENT_REDUCTION_TARGET = 0.20
# The spec does not quantify "severe": a stem regresses severely when its mean
# per-stem onset+pitch F1 drops by more than 0.05 against warm 'off'.
SEVERE_STEM_F1_DROP = 0.05
EXTRA_TIME_LIMIT = 0.15
EXTRA_RAM_LIMIT_MB = 512.0
SDR_TOLERANCE_DB = 0.01
_EPS = 1e-9


def _per_stem_f1(workspace, mix_path, ground_truth_midi) -> Dict[str, float]:
    """Onset+pitch F1 per estimated stem against the same-named reference group."""
    import pretty_midi

    reference = pretty_midi.PrettyMIDI(str(ground_truth_midi))
    result = {}
    groups = {inst.name for inst in reference.instruments}
    for role, path in active_midis(workspace.get_manifest(Path(mix_path).resolve())).items():
        if role not in groups:
            continue
        ref_notes = [n for inst in reference.instruments if inst.name == role for n in inst.notes]
        est_notes = [n for inst in pretty_midi.PrettyMIDI(str(path)).instruments for n in inst.notes]
        result[role] = score_notes(ref_notes, est_notes)["f1"]
    return result


def worker_main(spec: dict) -> dict:
    """One pipeline+evaluate run, meant to execute in a fresh process."""
    global run_pipeline, evaluate_quality
    stub = os.environ.get("MUSIC_RE_BENCH_STUB")
    if stub:
        module = importlib.import_module(stub)
        run_pipeline, evaluate_quality = module.run_pipeline, module.evaluate_quality
    ws = WorkspaceManager(cache_root=spec["cache_root"])
    quality_report, error, wall_time_sec, peak_rss_mb = _run_one(
        spec["mix_path"], ws, ground_truth_stems=spec.get("ground_truth_stems"),
        ground_truth_midi=spec.get("ground_truth_midi"), profile=spec.get("profile", "fast"),
        six_stems=spec.get("six_stems", False), spectral_validation=spec["mode"])
    per_stem, per_stem_error = {}, None
    if spec.get("six_stems"):
        # Slakh references group guitar/piano into "other"; six-stem output has no matching reference.
        per_stem, per_stem_error = None, "six-stem output not scored: references group guitar/piano into other"
    elif error is None and spec.get("ground_truth_midi"):
        try:
            per_stem = _per_stem_f1(ws, spec["mix_path"], spec["ground_truth_midi"])
        except Exception as e:
            per_stem_error = str(e)
    return dict(quality_report=quality_report, wall_time_sec=wall_time_sec, peak_rss_mb=peak_rss_mb,
                error=error, per_stem_f1=per_stem, per_stem_error=per_stem_error)


def _run_worker(spec: dict) -> dict:
    """Run worker_main in a child process so its peak RSS is its own."""
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", json.dumps(spec)],
                          capture_output=True, text=True)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(WORKER_RESULT_PREFIX):
            return json.loads(line[len(WORKER_RESULT_PREFIX):])
    return dict(quality_report=None, wall_time_sec=None, peak_rss_mb=None, per_stem_f1={},
                error=f"worker exited {proc.returncode}: {proc.stderr.strip()[-2000:]}")


def run_spectral_ab(prepared_tracks: List[dict], cache_root: Path, modes: List[str], profile: str = "fast",
                    six_stems: bool = False, runner=None) -> List[dict]:
    """Cold 'off', warm 'off', then each other mode, all on one cache root per track."""
    runner = runner or _run_worker
    sequence = [("off_cold", "off"), ("off", "off")] + [(m, m) for m in modes if m != "off"]
    results = []
    for track in prepared_tracks:
        track_root = Path(cache_root) / track["track_id"]
        group_status = dict(track.get("group_status", {}))
        runs = {}
        for name, mode in sequence:
            run = runner(dict(mix_path=track["mix_path"], cache_root=str(track_root), mode=mode, profile=profile,
                              six_stems=six_stems, ground_truth_stems=track["ground_truth_stems_dir"],
                              ground_truth_midi=track["ground_truth_midi_path"]))
            run["false_notes_on_absent_instrument"] = _false_notes_on_absent_instrument(
                run.get("quality_report"), group_status)
            runs[name] = run
        results.append(dict(track_id=track["track_id"], group_status=group_status, cache_root=str(track_root),
                            preparation_warnings=list(track.get("warnings", [])), runs=runs))
    return results


def _pitched(run: dict, key: str):
    return (((run.get("quality_report") or {}).get("transcription_metrics") or {}).get("pitched") or {}).get(
        f"onset_pitch_{key}")


def _sdr_values(run: dict) -> Dict[str, float]:
    sep = (run.get("quality_report") or {}).get("separation_metrics") or {}
    values = {f"stem:{k}": v for k, v in (sep.get("sdr") or {}).items() if v is not None}
    if sep.get("average_sdr_db") is not None:
        values["average"] = sep["average_sdr_db"]
    return values


def _ok(run: Optional[dict]) -> bool:
    return bool(run) and run.get("error") is None and run.get("quality_report") is not None


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _absent_total(run: dict, group_status: dict) -> Optional[int]:
    """Notes on absent groups; None unless every absent group has a measured count."""
    stems = ((run.get("quality_report") or {}).get("transcription_metrics") or {}).get("stems") or {}
    counts = [(stems.get(g) or {}).get("notes_count") for g, s in group_status.items() if s == ABSENT]
    return None if None in counts else sum(counts)


def _sum_or_none(values):
    values = list(values)
    return None if None in values else sum(values)


def _mode_metrics(tracks: List[dict], name: str) -> dict:
    runs = [(t, t["runs"].get(name)) for t in tracks]
    ok = [(t, r) for t, r in runs if _ok(r)]
    stems = sorted({s for _, r in ok for s in (r.get("per_stem_f1") or {})})
    return dict(
        n_ok=len(ok), n_errors=len(runs) - len(ok),
        **{f"pitched_{k}": _mean(_pitched(r, k) for _, r in ok) for k in ("precision", "recall", "f1")},
        per_stem_f1={s: _mean((r.get("per_stem_f1") or {}).get(s) for _, r in ok) for s in stems},
        absent_false_notes=_sum_or_none(_absent_total(r, t["group_status"]) for t, r in ok) if ok else None,
        wall_time_sec=_mean(r.get("wall_time_sec") for _, r in ok),
        peak_rss_mb_max=max((r["peak_rss_mb"] for _, r in ok if r.get("peak_rss_mb") is not None), default=None))


def _criterion(value, passed, target):
    if value is None:
        return dict(status="NOT_MEASURED", value=None, target=target)
    return dict(status="PASS" if passed else "FAIL", value=value, target=target)


def _compare(tracks: List[dict], mode: str, reused_cache: bool = False) -> dict:
    paired = [t for t in tracks if _ok(t["runs"].get("off")) and _ok(t["runs"].get(mode))]
    per_track, deltas = [], {k: [] for k in ("precision", "recall", "f1")}
    stem_deltas: Dict[str, list] = {}
    absent_off = absent_mode = 0
    absent_measured = absent_missing = per_stem_unscored = False
    extra_times, cold_times, extra_rams, sdr_deltas = [], [], [], []
    for t in paired:
        off, trial, cold = t["runs"]["off"], t["runs"][mode], t["runs"].get("off_cold")
        row = dict(track_id=t["track_id"])
        for k in deltas:
            a, b = _pitched(off, k), _pitched(trial, k)
            row[f"delta_{k}"] = None if a is None or b is None else b - a
            if row[f"delta_{k}"] is not None:
                deltas[k].append(row[f"delta_{k}"])
        if off.get("per_stem_f1") is None or trial.get("per_stem_f1") is None:
            per_stem_unscored = True
        for stem, b in (trial.get("per_stem_f1") or {}).items():
            a = (off.get("per_stem_f1") or {}).get(stem)
            if a is not None and b is not None:
                stem_deltas.setdefault(stem, []).append(b - a)
        if any(s == ABSENT for s in t["group_status"].values()):
            a, b = _absent_total(off, t["group_status"]), _absent_total(trial, t["group_status"])
            if a is None or b is None:
                absent_missing = True
            else:
                absent_measured = True
                absent_off, absent_mode = absent_off + a, absent_mode + b
        row["extra_time_sec"] = row["extra_ram_mb"] = None
        if _ok(cold) and None not in (cold.get("wall_time_sec"), off.get("wall_time_sec"), trial.get("wall_time_sec")):
            row["extra_time_sec"] = trial["wall_time_sec"] - off["wall_time_sec"]
            extra_times.append(row["extra_time_sec"])
            cold_times.append(cold["wall_time_sec"])
        if off.get("peak_rss_mb") is not None and trial.get("peak_rss_mb") is not None:
            row["extra_ram_mb"] = trial["peak_rss_mb"] - off["peak_rss_mb"]
            extra_rams.append(row["extra_ram_mb"])
        a_sdr, b_sdr = _sdr_values(off), _sdr_values(trial)
        sdr_deltas += [abs(b_sdr[k] - a_sdr[k]) for k in a_sdr if k in b_sdr]
        per_track.append(row)

    f1 = deltas["f1"]
    distribution = dict(min=min(f1), median=statistics.median(f1), max=max(f1), n=len(f1),
                        recall_drops_over_0_02=sum(d < -RECALL_DROP_LIMIT - _EPS for d in deltas["recall"])) if f1 else None
    mean = {k: _mean(v) for k, v in deltas.items()}
    stem_means = {s: statistics.mean(v) for s, v in stem_deltas.items()}
    worst_stem = min(stem_means, key=stem_means.get) if stem_means and not per_stem_unscored else None
    reduction = ((absent_off - absent_mode) / absent_off
                 if absent_measured and absent_off and not absent_missing else None)
    time_ratio = sum(extra_times) / sum(cold_times) if cold_times and sum(cold_times) > 0 else None
    ram = max(extra_rams) if extra_rams else None
    sdr = max(sdr_deltas) if sdr_deltas else None
    n_errors = sum(not _ok(t["runs"].get(mode)) for t in tracks)
    acceptance = dict(
        f1=_criterion(mean["f1"], mean["f1"] is not None and mean["f1"] >= -_EPS, "mean ΔF1 >= 0"),
        precision=_criterion(mean["precision"], mean["precision"] is not None and mean["precision"] >= -_EPS,
                             "mean Δprecision >= 0"),
        recall=_criterion(mean["recall"], mean["recall"] is not None and mean["recall"] >= -RECALL_DROP_LIMIT - _EPS,
                          "mean Δrecall >= -0.02"),
        absent_false_notes=_criterion(reduction, reduction is not None and reduction >= ABSENT_REDUCTION_TARGET - _EPS,
                                      "absent-instrument false notes reduced >= 20%"),
        per_stem=_criterion(stem_means[worst_stem] if worst_stem else None,
                            worst_stem is not None and stem_means[worst_stem] >= -SEVERE_STEM_F1_DROP - _EPS,
                            "no stem mean ΔF1 < -0.05"),
        extra_time=_criterion(time_ratio, time_ratio is not None and time_ratio <= EXTRA_TIME_LIMIT + _EPS,
                              "Σ(T_mode_warm - T_off_warm) / ΣT_off_cold <= 0.15"),
        extra_ram=_criterion(ram, ram is not None and ram <= EXTRA_RAM_LIMIT_MB + _EPS,
                             "max per-track RSS(mode) - RSS(warm off) <= 512 MiB"),
        sdr=_criterion(sdr, sdr is not None and sdr < SDR_TOLERANCE_DB, "max |ΔSDR| < 0.01 dB"),
        errors=_criterion(n_errors, n_errors == 0, "no failed runs"),
    )
    if worst_stem:
        acceptance["per_stem"]["stem"] = worst_stem
    if reused_cache:
        acceptance["extra_time"] = _criterion(None, False, "NOT_MEASURED: reused cache is not cold")
    return dict(n_paired=len(paired), delta=dict(**{f"pitched_{k}": v for k, v in mean.items()},
                                                  per_stem_f1=stem_means, absent_false_notes_reduction=reduction),
                f1_distribution=distribution, per_track=per_track, acceptance=acceptance)


def summarize_spectral_ab(tracks: List[dict], modes: List[str], reused_cache: bool = False) -> dict:
    names = ["off_cold", "off"] + [m for m in modes if m != "off"]
    return dict(modes={n: _mode_metrics(tracks, n) for n in names},
                comparisons={m: _compare(tracks, m, reused_cache) for m in modes if m != "off"})


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _fmt_drum_class(class_metrics) -> str:
    """f1 alone can't tell "the model found nothing" (ref_count=0) apart
    from "the model found events, all wrong" (ref_count>0, est_count>0,
    zero overlap) -- show the raw counts beside the derived score."""
    if not class_metrics:
        return "-"
    return (f"{_fmt(class_metrics.get('f1'))} "
            f"(ref={class_metrics.get('ref_count', '?')}, est={class_metrics.get('est_count', '?')})")


def discover_benchmark_tracks(raw_dir: Path) -> list[Path]:
    """Discover track directories under raw_dir, ignoring hidden/staging entries."""
    if not raw_dir.is_dir():
        return []
    return sorted(d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith("."))


def render_markdown_report(report: dict) -> str:
    lines = ["# BenchmarkReport", "", f"Git revision: `{report.get('git_revision', 'unknown')}`", ""]
    if report.get("reused_cache"):
        lines += [f"Reused cache: `{report['reused_cache']}`", ""]

    lines += ["## Reference tracks (Slakh ground truth)", "",
              "| Track | SDR (dB) | Pitched F1 | Drums F1 | Kick F1 | Snare F1 | Hihat F1 | "
              "False notes (absent instrument) | Wall time (s) | Peak RAM (MB) | Error |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for entry in report.get("reference_tracks", []):
        qr = entry.get("quality_report") or {}
        sep = qr.get("separation_metrics") or {}
        trans = qr.get("transcription_metrics") or {}
        by_class = trans.get("drums_by_class") or {}
        false_notes = entry.get("false_notes_on_absent_instrument") or {}
        avg_sdr = _fmt(sep.get("average_sdr_db"))
        if sep.get("average_sdr_count") is not None:
            avg_sdr = f"{avg_sdr} (n={sep.get('average_sdr_count')})"
        row = [
            entry["track_id"],
            avg_sdr,
            _fmt((trans.get("pitched") or {}).get("onset_pitch_f1")),
            _fmt((trans.get("drums") or {}).get("onset_pitch_f1")),
            _fmt_drum_class(by_class.get("kick")),
            _fmt_drum_class(by_class.get("snare")),
            _fmt_drum_class(by_class.get("hihat")),
            ", ".join(f"{k}={v}" for k, v in false_notes.items()) or "-",
            _fmt(entry.get("wall_time_sec")),
            _fmt(entry.get("peak_ram_mb")),
            entry.get("error") or "-",
        ]
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

    lines += ["", "## Per-stem separation (SDR)", "",
              "An average SDR hides whether every instrument separated equally well.",
              "", "| Track | Stem | SDR (dB) |", "|---|---|---|"]
    for entry in report.get("reference_tracks", []):
        sep = (entry.get("quality_report") or {}).get("separation_metrics") or {}
        stem_statuses = sep.get("stem_status") or {}
        sdr_dict = sep.get("sdr") or {}
        all_stems = sorted(set(list(sdr_dict.keys()) + list(stem_statuses.keys())))
        for stem in all_stems:
            val = sdr_dict.get(stem)
            status = stem_statuses.get(stem)
            display_val = _fmt(val) if val is not None else (status or "-")
            if stem == "other" and sep.get("other_includes"):
                display_stem = f"other (includes {', '.join(s for s in sep['other_includes'] if s != 'other')})"
            else:
                display_stem = stem
            lines.append(f"| {entry['track_id']} | {display_stem} | {display_val} |")

    lines += ["", "## Reference completeness", "",
              "Only `absent` groups count toward false notes on an absent instrument; "
              "`missing_annotation` (audio, no MIDI) and `not_rendered` (MIDI, no audio) "
              "mean the reference itself is incomplete.",
              "", "| Track | vocals | drums | bass | other |", "|---|---|---|---|---|"]
    for entry in report.get("reference_tracks", []):
        status = entry.get("group_status") or {}
        lines.append(f"| {entry['track_id']} | " + " | ".join(
            status.get(g, "-") for g in ("vocals", "drums", "bass", "other")) + " |")
    warning_lines = [f"- `{entry['track_id']}`: {w}"
                     for entry in report.get("reference_tracks", [])
                     for w in entry.get("preparation_warnings") or []]
    lines += ["", "### Preparation warnings", ""] + (warning_lines or ["_None._"])

    lines += ["", "## Real-track spot checks (no ground truth \u2014 manual review)", ""]
    spotcheck = report.get("manual_spotcheck") or []
    if not spotcheck:
        lines.append("_None run._")
    else:
        for entry in spotcheck:
            qr = entry.get("quality_report") or {}
            trans = qr.get("transcription_metrics") or {}
            lines.append(f"- `{entry['track_id']}`: {trans.get('total_notes', '?')} notes transcribed, "
                          f"{len(trans.get('warnings') or [])} warnings, error={entry.get('error') or 'none'}")

    if report.get("spectral_validation"):
        lines += _render_spectral_section(report["spectral_validation"])
    return "\n".join(lines)


def _render_spectral_section(sv: dict) -> List[str]:
    lines = ["", "## Spectral validation A/B", "",
             f"Modes: {', '.join(sv.get('modes_requested', []))}; split: {sv.get('split', 'all')}. "
             "'off' is the warm rerun on the same cache; 'off_cold' is the first full run (T_off).", "",
             "| Mode | OK | Errors | Pitched P | Pitched R | Pitched F1 | Per-stem F1 | "
             "False notes (absent) | Mean wall time (s) | Max peak RSS (MiB) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in (sv.get("modes") or {}).items():
        per_stem = ", ".join(f"{k}={_fmt(v)}" for k, v in (m.get("per_stem_f1") or {}).items()) or "-"
        lines.append("| " + " | ".join(str(c) for c in [
            name, m.get("n_ok"), m.get("n_errors"), _fmt(m.get("pitched_precision")), _fmt(m.get("pitched_recall")),
            _fmt(m.get("pitched_f1")), per_stem, _fmt(m.get("absent_false_notes")), _fmt(m.get("wall_time_sec")),
            _fmt(m.get("peak_rss_mb_max"))]) + " |")
    for mode, comp in (sv.get("comparisons") or {}).items():
        delta, dist = comp.get("delta") or {}, comp.get("f1_distribution")
        lines += ["", f"### {mode} vs off (paired tracks: {comp.get('n_paired')})", "",
                  f"Mean deltas: precision {_fmt(delta.get('pitched_precision'))}, "
                  f"recall {_fmt(delta.get('pitched_recall'))}, F1 {_fmt(delta.get('pitched_f1'))}; "
                  f"absent false-note reduction {_fmt(delta.get('absent_false_notes_reduction'))}.", ""]
        lines.append("Per-track ΔF1: " + (f"min {_fmt(dist['min'])}, median {_fmt(dist['median'])}, "
                                         f"max {_fmt(dist['max'])} (n={dist['n']}); tracks with recall drop "
                                         f"> 0.02: {dist['recall_drops_over_0_02']}" if dist else "NOT_MEASURED"))
        lines += ["", "| Criterion | Target | Measured | Status |", "|---|---|---|---|"]
        for name, c in (comp.get("acceptance") or {}).items():
            measured = _fmt(c.get("value")) + (f" ({c['stem']})" if c.get("stem") else "")
            lines.append(f"| {name} | {c.get('target')} | {measured} | {c.get('status')} |")
        lines += ["", "| Track | ΔP | ΔR | ΔF1 | Extra time (s) | Extra RSS (MiB) |", "|---|---|---|---|---|---|"]
        for row in comp.get("per_track") or []:
            lines.append("| " + " | ".join(str(c) for c in [
                row["track_id"], _fmt(row.get("delta_precision")), _fmt(row.get("delta_recall")),
                _fmt(row.get("delta_f1")), _fmt(row.get("extra_time_sec")), _fmt(row.get("extra_ram_mb"))]) + " |")
    return lines


def _git_revision() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        rev = result.stdout.strip()
        return rev if result.returncode == 0 and rev else "unknown"
    except Exception:
        return "unknown"


def _atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp.name)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MusicRE Slakh benchmark baseline.")
    parser.add_argument("--slakh-raw-dir", default="data/benchmarks/slakh_raw/raw")
    parser.add_argument("--prepared-dir", default="data/benchmarks/slakh_prepared")
    parser.add_argument("--real-track", action="append", default=[])
    parser.add_argument("--out-json", default="BenchmarkReport.json")
    parser.add_argument("--out-md", default="BenchmarkReport.md")
    parser.add_argument("--history-dir", default="data/benchmarks/history")
    parser.add_argument("--profile", default="fast", choices=["fast", "balanced", "quality"])
    parser.add_argument("--six-stems", action="store_true", default=False)
    parser.add_argument("--spectral-validation", nargs="+", choices=["off", "audit", "filter"], default=["off"])
    parser.add_argument("--split", choices=["tune", "test", "all"], default="all")
    parser.add_argument("--reuse-cache", metavar="PATH", help="Reuse existing benchmark cache root")
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    return parser


def _reference_entry(track: dict) -> dict:
    run = track["runs"]["off_cold"]
    return dict(track_id=track["track_id"], quality_report=run.get("quality_report"),
                wall_time_sec=run.get("wall_time_sec"), peak_ram_mb=run.get("peak_rss_mb"), error=run.get("error"),
                false_notes_on_absent_instrument=run.get("false_notes_on_absent_instrument", {}),
                group_status=track["group_status"], preparation_warnings=track["preparation_warnings"])


def main():
    args = build_parser().parse_args()
    if args._worker:
        print(WORKER_RESULT_PREFIX + json.dumps(worker_main(json.loads(args._worker)), default=str), flush=True)
        return

    raw_dir = Path(args.slakh_raw_dir)
    prepared_dir = Path(args.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    track_dirs = discover_benchmark_tracks(raw_dir)
    if not track_dirs:
        track_dirs = fetch_babyslakh(raw_dir.parent)
    track_dirs = [d for d in track_dirs if track_in_split(Path(d).name, args.split)]

    # prepare_track is cheap and deterministic; re-running it keeps
    # group_note_counts correct without a separate "already prepared" cache.
    prepared_tracks, prep_failures = prepare_all_tracks(track_dirs, prepared_dir)

    cache_root = Path(args.reuse_cache) if args.reuse_cache else _fresh_cache_root(Path("data/benchmarks/cache"))
    modes = list(dict.fromkeys(args.spectral_validation))
    spectral = None
    if modes == ["off"]:
        reference_results = run_reference_benchmarks(prepared_tracks, cache_root=cache_root, profile=args.profile, six_stems=args.six_stems) + prep_failures
    else:
        ab_tracks = run_spectral_ab(prepared_tracks, cache_root, modes, profile=args.profile, six_stems=args.six_stems)
        reference_results = [_reference_entry(t) for t in ab_tracks] + prep_failures
        spectral = dict(modes_requested=modes, split=args.split, tracks=ab_tracks,
                        **summarize_spectral_ab(ab_tracks, modes, reused_cache=bool(args.reuse_cache)))
    manual_results = run_manual_spotcheck([Path(p) for p in args.real_track], cache_root=cache_root, profile=args.profile, six_stems=args.six_stems)

    report = dict(
        reference_tracks=reference_results,
        manual_spotcheck=manual_results,
        git_revision=_git_revision(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        profile=args.profile,
        six_stems=args.six_stems,
        reused_cache=str(cache_root) if args.reuse_cache else None,
    )
    if spectral is not None:
        report["spectral_validation"] = spectral

    _atomic_write_text(Path(args.out_json), json.dumps(report, indent=2))
    _atomic_write_text(Path(args.out_md), render_markdown_report(report))

    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copyfile(args.out_json, history_dir / f"{timestamp}_{report['git_revision']}.json")

    print(f"Wrote {args.out_json}, {args.out_md}, and a history snapshot under {history_dir}/")


if __name__ == "__main__":
    main()
