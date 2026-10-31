#!/usr/bin/env python3
"""Orchestrate the Slakh benchmark baseline: run MusicRE's real pipeline
against a prepared Slakh subset (objective, mir_eval/museval-scored) and,
optionally, a few real tracks with no ground truth (manual spot check).
Writes BenchmarkReport.json / BenchmarkReport.md and a timestamped history
snapshot so later versions can be compared against this one.
"""
import argparse
import hashlib
import json
import resource
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from music_re.core.pipeline import run_pipeline
from music_re.core.quality import evaluate_quality
from music_re.workspace import WorkspaceManager
from scripts.fetch_slakh_subset import fetch_babyslakh
from scripts.slakh_mapping import ABSENT, prepare_track


def track_in_split(track_id: str, split: str) -> bool:
    if split not in ('all', 'tune', 'test'):
        raise ValueError(f'Invalid split: {split}')
    return split == 'all' or (int(hashlib.sha1(track_id.encode()).hexdigest(), 16) % 2 == (split == 'test'))


def compare_modes(baseline: List[dict], trial: List[dict]) -> dict:
    by_id = {entry['track_id']: entry for entry in baseline}
    f1_deltas, recall_deltas = [], []
    for entry in trial:
        original = by_id.get(entry['track_id'])
        if original is None or not original.get('quality_report') or not entry.get('quality_report'):
            continue
        a = (original['quality_report'].get('transcription_metrics') or {}).get('pitched') or {}
        b = (entry['quality_report'].get('transcription_metrics') or {}).get('pitched') or {}
        for key, result in [('onset_pitch_f1', f1_deltas), ('onset_pitch_recall', recall_deltas)]:
            if a.get(key) is not None and b.get(key) is not None:
                result.append(b[key] - a[key])
    distribution = {'min': min(f1_deltas), 'median': statistics.median(f1_deltas),
                    'max': max(f1_deltas), 'drops_over_0_02': sum(x < -0.02 - 1e-12 for x in f1_deltas)} if f1_deltas else None
    recall = statistics.mean(recall_deltas) if recall_deltas else None
    return {'f1_distribution': distribution, 'acceptance': {'recall': {
        'value': recall, 'status': 'NOT EVALUATED' if recall is None else ('PASS' if recall >= -0.02 - 1e-12 else 'FAIL')}}}


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


def _run_one(mix_path, workspace, ground_truth_stems=None, ground_truth_midi=None, profile="fast", six_stems=False):
    """Time one pipeline+evaluate run; never raises, reports error instead."""
    t0 = time.perf_counter()
    quality_report, error = None, None
    try:
        run_pipeline(mix_path, workspace=workspace, profile=profile, six_stems=six_stems)
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

    return "\n".join(lines)


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


def main():
    parser = argparse.ArgumentParser(description="Run the MusicRE Slakh benchmark baseline.")
    parser.add_argument("--slakh-raw-dir", default="data/benchmarks/slakh_raw/raw")
    parser.add_argument("--prepared-dir", default="data/benchmarks/slakh_prepared")
    parser.add_argument("--real-track", action="append", default=[])
    parser.add_argument("--out-json", default="BenchmarkReport.json")
    parser.add_argument("--out-md", default="BenchmarkReport.md")
    parser.add_argument("--history-dir", default="data/benchmarks/history")
    parser.add_argument("--profile", default="fast", choices=["fast", "balanced", "quality"])
    parser.add_argument("--six-stems", action="store_true", default=False)
    args = parser.parse_args()

    raw_dir = Path(args.slakh_raw_dir)
    prepared_dir = Path(args.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    track_dirs = discover_benchmark_tracks(raw_dir)
    if not track_dirs:
        track_dirs = fetch_babyslakh(raw_dir.parent)

    # prepare_track is cheap and deterministic; re-running it keeps
    # group_note_counts correct without a separate "already prepared" cache.
    prepared_tracks, prep_failures = prepare_all_tracks(track_dirs, prepared_dir)

    cache_root = _fresh_cache_root(Path("data/benchmarks/cache"))
    reference_results = run_reference_benchmarks(prepared_tracks, cache_root=cache_root, profile=args.profile, six_stems=args.six_stems) + prep_failures
    manual_results = run_manual_spotcheck([Path(p) for p in args.real_track], cache_root=cache_root, profile=args.profile, six_stems=args.six_stems)

    report = dict(
        reference_tracks=reference_results,
        manual_spotcheck=manual_results,
        git_revision=_git_revision(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        profile=args.profile,
        six_stems=args.six_stems,
    )

    _atomic_write_text(Path(args.out_json), json.dumps(report, indent=2))
    _atomic_write_text(Path(args.out_md), render_markdown_report(report))

    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copyfile(args.out_json, history_dir / f"{timestamp}_{report['git_revision']}.json")

    print(f"Wrote {args.out_json}, {args.out_md}, and a history snapshot under {history_dir}/")


if __name__ == "__main__":
    main()
