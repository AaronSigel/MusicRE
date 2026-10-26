#!/usr/bin/env python3
"""Orchestrate the Slakh benchmark baseline: run MusicRE's real pipeline
against a prepared Slakh subset (objective, mir_eval/museval-scored) and,
optionally, a few real tracks with no ground truth (manual spot check).
Writes BenchmarkReport.json / BenchmarkReport.md and a timestamped history
snapshot so later versions can be compared against this one.
"""
import argparse
import json
import resource
import shutil
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
from scripts.slakh_mapping import prepare_track


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


def _run_one(mix_path, workspace, ground_truth_stems=None, ground_truth_midi=None):
    """Time one pipeline+evaluate run; never raises, reports error instead."""
    t0 = time.perf_counter()
    quality_report, error = None, None
    try:
        run_pipeline(mix_path, workspace=workspace)
        report = evaluate_quality(workspace, mix_path, ground_truth_stems=ground_truth_stems,
                                   ground_truth_midi=ground_truth_midi)
        quality_report = _as_dict(report)
    except Exception as e:
        error = str(e)
    wall_time_sec = time.perf_counter() - t0
    return quality_report, error, wall_time_sec, _peak_rss_mb()


def _false_notes_on_absent_instrument(quality_report: Optional[dict], group_note_counts: Dict[str, int]) -> Dict[str, int]:
    if quality_report is None:
        return {}
    stems = (quality_report.get("transcription_metrics") or {}).get("stems", {})
    return {group: stems[group]["notes_count"]
            for group, count in group_note_counts.items()
            if count == 0 and group in stems}


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
                                  peak_ram_mb=0.0, error=str(e), false_notes_on_absent_instrument={}))
    return prepared, failures


def run_reference_benchmarks(prepared_tracks: List[dict], cache_root: Path) -> List[dict]:
    results = []
    for track in prepared_tracks:
        track_id = track["track_id"]
        ws = WorkspaceManager(cache_root=Path(cache_root) / track_id)
        quality_report, error, wall_time_sec, peak_ram_mb = _run_one(
            track["mix_path"], ws,
            ground_truth_stems=track["ground_truth_stems_dir"],
            ground_truth_midi=track["ground_truth_midi_path"],
        )
        results.append(dict(
            track_id=track_id, quality_report=quality_report, wall_time_sec=wall_time_sec,
            peak_ram_mb=peak_ram_mb, error=error,
            false_notes_on_absent_instrument=_false_notes_on_absent_instrument(
                quality_report, track.get("group_note_counts", {})),
        ))
    return results


def run_manual_spotcheck(real_tracks: List[Path], cache_root: Path) -> List[dict]:
    results = []
    for path in real_tracks:
        path = Path(path)
        track_id = path.stem
        ws = WorkspaceManager(cache_root=Path(cache_root) / track_id)
        quality_report, error, wall_time_sec, peak_ram_mb = _run_one(str(path), ws)
        results.append(dict(track_id=track_id, quality_report=quality_report, wall_time_sec=wall_time_sec,
                             peak_ram_mb=peak_ram_mb, error=error))
    return results


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


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
        row = [
            entry["track_id"],
            _fmt(sep.get("average_sdr_db")),
            _fmt((trans.get("pitched") or {}).get("onset_pitch_f1")),
            _fmt((trans.get("drums") or {}).get("onset_pitch_f1")),
            _fmt((by_class.get("kick") or {}).get("f1")),
            _fmt((by_class.get("snare") or {}).get("f1")),
            _fmt((by_class.get("hihat") or {}).get("f1")),
            ", ".join(f"{k}={v}" for k, v in false_notes.items()) or "-",
            _fmt(entry.get("wall_time_sec")),
            _fmt(entry.get("peak_ram_mb")),
            entry.get("error") or "-",
        ]
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

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


def main():
    parser = argparse.ArgumentParser(description="Run the MusicRE Slakh benchmark baseline.")
    parser.add_argument("--slakh-raw-dir", default="data/benchmarks/slakh_raw/raw")
    parser.add_argument("--prepared-dir", default="data/benchmarks/slakh_prepared")
    parser.add_argument("--real-track", action="append", default=[])
    parser.add_argument("--out-json", default="BenchmarkReport.json")
    parser.add_argument("--out-md", default="BenchmarkReport.md")
    parser.add_argument("--history-dir", default="data/benchmarks/history")
    args = parser.parse_args()

    raw_dir = Path(args.slakh_raw_dir)
    prepared_dir = Path(args.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    track_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir()) if raw_dir.is_dir() else []
    if not track_dirs:
        track_dirs = fetch_babyslakh(raw_dir.parent)

    # prepare_track is cheap and deterministic; re-running it keeps
    # group_note_counts correct without a separate "already prepared" cache.
    prepared_tracks, prep_failures = prepare_all_tracks(track_dirs, prepared_dir)

    cache_root = _fresh_cache_root(Path("data/benchmarks/cache"))
    reference_results = run_reference_benchmarks(prepared_tracks, cache_root=cache_root) + prep_failures
    manual_results = run_manual_spotcheck([Path(p) for p in args.real_track], cache_root=cache_root)

    report = dict(
        reference_tracks=reference_results,
        manual_spotcheck=manual_results,
        git_revision=_git_revision(),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    Path(args.out_json).write_text(json.dumps(report, indent=2))
    Path(args.out_md).write_text(render_markdown_report(report))

    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copyfile(args.out_json, history_dir / f"{timestamp}_{report['git_revision']}.json")

    print(f"Wrote {args.out_json}, {args.out_md}, and a history snapshot under {history_dir}/")


if __name__ == "__main__":
    main()
