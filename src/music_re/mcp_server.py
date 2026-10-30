import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP

from music_re.core.analyzer import analyze_audio as core_analyze_audio
from music_re.core.backends import get_structure_analyzer, get_transcriber_backend
from music_re.core.export import export_bundle as core_export_bundle
from music_re.core.midi_digest import generate_midi_digest as core_generate_midi_digest
from music_re.core.midi_drums import transcribe_drums as core_transcribe_drums
from music_re.core.midi_project import build_combined_midi_project
from music_re.core.midi_tonal import transcribe_tonal as core_transcribe_tonal
from music_re.core.quality import evaluate_quality as core_evaluate_quality
from music_re.core.separator import separate_stems as core_separate_stems
from music_re.core.slicer import slice_audio as core_slice_audio
from music_re.core.pipeline import run_pipeline
from music_re.workspace import WorkspaceManager

mcp = FastMCP("music-re")


@mcp.tool()
async def slice_audio(file_path: str, start_sec: float, end_sec: float) -> str:
    """Extract a segment from an audio file to save compute time and memory."""
    in_p = Path(file_path).resolve()
    start_str = f"{start_sec:g}"
    end_str = f"{end_sec:g}"
    out_p = in_p.with_name(f"{in_p.stem}_slice_{start_str}_{end_str}{in_p.suffix}")
    res = await asyncio.to_thread(core_slice_audio, in_p, out_p, start_sec=start_sec, end_sec=end_sec)
    return str(res)


@mcp.tool()
async def analyze_track(file_path: str) -> Dict[str, Any]:
    """Analyze tempo (BPM), musical key, beat grid, and integrated loudness (LUFS)."""
    p = Path(file_path).resolve()
    analysis = await asyncio.to_thread(core_analyze_audio, p)
    return analysis.model_dump()


@mcp.tool()
async def separate_stems(
    file_path: str,
    six_stems: bool = False,
    device: str = "auto",
    cache_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Separate audio into stems (vocals, drums, bass, other; + guitar, piano with six_stems)."""
    p = Path(file_path).resolve()

    def _run() -> Dict[str, Any]:
        ws = WorkspaceManager(cache_root=cache_root)
        result = core_separate_stems(p, workspace=ws, six_stems=six_stems, device=device)
        return result.model_dump()

    return await asyncio.to_thread(_run)


@mcp.tool()
async def transcribe_stem(
    stem_path: str,
    mode: str = "tonal",
    bpm: float = 120.0,
    timeline_path: Optional[str] = None,
    grid: int = 4,
    quantize_ends: bool = True,
) -> Dict[str, Any]:
    """Transcribe an isolated audio stem to MIDI (mode: 'tonal' or 'drums') with optional dynamic timeline snapping."""
    p = Path(stem_path).resolve()
    out_mid = p.with_suffix(".mid")

    def _run() -> Dict[str, Any]:
        tl_obj = None
        if timeline_path:
            tl_p = Path(timeline_path)
            if not tl_p.is_file():
                raise FileNotFoundError(f"Timeline file not found: {timeline_path}")
            from music_re.schemas import MusicalTimeline
            with open(tl_p, "r", encoding="utf-8") as f:
                tl_obj = MusicalTimeline(**json.load(f))

        transcriber = get_transcriber_backend(mode=mode)
        res = transcriber.transcribe(
            audio_path=p,
            output_midi_path=out_mid,
            timeline=tl_obj,
            bpm=bpm,
            quantize=True,
            save_raw=True,
            grid_subdivisions=grid,
            quantize_ends=quantize_ends,
        )
        return res.model_dump()

    return await asyncio.to_thread(_run)


@mcp.tool()
async def inspect_midi_bars(
    midi_path: str,
    start_bar: int = 1,
    end_bar: int = 8,
    bpm: float = 120.0,
) -> Dict[str, Any]:
    """Read MIDI notes in a bar-by-bar, beat-indexed format optimized for LLM context."""
    p = Path(midi_path).resolve()
    digest = await asyncio.to_thread(
        core_generate_midi_digest, p, bpm=bpm, start_bar=start_bar, end_bar=end_bar
    )
    return digest.model_dump()


@mcp.tool()
async def get_structure_timeline(
    file_path: str,
    backend: str = "auto",
    cache_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract musical timeline (sections, beats, downbeats, BPM) into timeline.json."""
    p = Path(file_path).resolve()

    def _run() -> Dict[str, Any]:
        ws = WorkspaceManager(cache_root=cache_root)
        with ws.lock(p):
            manifest = ws.get_manifest(p)
            analyzer = get_structure_analyzer(name=backend)
            timeline = analyzer.analyze(p, stems=manifest.stems)
            ws.save_timeline(p, timeline)
            return timeline.model_dump()

    return await asyncio.to_thread(_run)


@mcp.tool()
async def evaluate_quality_report(
    file_path: str,
    ground_truth_stems: Optional[str] = None,
    ground_truth_midi: Optional[str] = None,
    cache_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate decomposition and transcription quality (objective if ground truth provided, diagnostic otherwise)."""
    p = Path(file_path).resolve()

    def _run() -> Dict[str, Any]:
        ws = WorkspaceManager(cache_root=cache_root)
        with ws.lock(p):
            report = core_evaluate_quality(
                workspace=ws,
                file_path=p,
                ground_truth_stems=ground_truth_stems,
                ground_truth_midi=ground_truth_midi,
            )
            ws.save_quality_report(p, report)
            return report.model_dump()

    return await asyncio.to_thread(_run)


@mcp.tool()
async def export_project_bundle(
    file_path: str,
    target_dir: str,
    profile: str = "balanced",
    archive: bool = False,
    cache_root: Optional[str] = None,
) -> str:
    """Export clean deliverable bundle containing WAV stems, MIDIs, timeline, manifest, and quality report."""
    p = Path(file_path).resolve()

    def _run() -> str:
        ws = WorkspaceManager(cache_root=cache_root)
        with ws.lock(p):
            res = core_export_bundle(
                workspace=ws,
                file_path=p,
                target_path=target_dir,
                profile=profile,
                archive=archive,
            )
            return str(res)

    return await asyncio.to_thread(_run)


@mcp.tool()
async def deconstruct_full_pipeline(
    file_path: str,
    six_stems: bool = False,
    device: str = "auto",
    profile: str = "balanced",
    cache_root: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end pipeline: separate stems, analyze acoustic features, extract timeline, transcribe to MIDI, and assemble project."""
    p = Path(file_path).resolve()

    def _run() -> Dict[str, Any]:
        ws = WorkspaceManager(cache_root=cache_root)
        with ws.lock(p):
            return run_pipeline(p, workspace=ws,
                                six_stems=six_stems, device=device, profile=profile)

    return await asyncio.to_thread(_run)


def main():
    """Main entrypoint for MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
