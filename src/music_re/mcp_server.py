from pathlib import Path
import sys
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP

from music_re.core.analyzer import analyze_audio as core_analyze_audio
from music_re.core.midi_digest import generate_midi_digest as core_generate_midi_digest
from music_re.core.midi_drums import transcribe_drums as core_transcribe_drums
from music_re.core.midi_tonal import transcribe_tonal as core_transcribe_tonal
from music_re.core.separator import separate_stems as core_separate_stems
from music_re.core.slicer import slice_audio as core_slice_audio
from music_re.workspace import WorkspaceManager

mcp = FastMCP("music-re")


@mcp.tool()
async def slice_audio(file_path: str, start_sec: float, end_sec: float) -> str:
    """Extract a segment from an audio file to save compute time and memory."""
    in_p = Path(file_path).resolve()
    out_p = in_p.with_name(f"{in_p.stem}_slice_{int(start_sec)}_{int(end_sec)}{in_p.suffix}")
    res = core_slice_audio(in_p, out_p, start_sec=start_sec, end_sec=end_sec)
    return str(res)


@mcp.tool()
async def analyze_track(file_path: str) -> Dict[str, Any]:
    """Analyze tempo (BPM), musical key, beat grid, and integrated loudness (LUFS)."""
    p = Path(file_path).resolve()
    analysis = core_analyze_audio(p)
    return analysis.model_dump()


@mcp.tool()
async def separate_stems(file_path: str, six_stems: bool = False) -> Dict[str, Any]:
    """Separate audio into stems (vocals, drums, bass, other; + guitar, piano with six_stems)."""
    p = Path(file_path).resolve()
    ws = WorkspaceManager()
    result = core_separate_stems(p, workspace=ws, six_stems=six_stems)
    return result.model_dump()


@mcp.tool()
async def transcribe_stem(stem_path: str, mode: str = "tonal", bpm: float = 120.0) -> str:
    """Transcribe an isolated audio stem to MIDI (mode: 'tonal' or 'drums')."""
    p = Path(stem_path).resolve()
    out_mid = p.with_suffix(".mid")
    if mode.lower() == "drums":
        res = core_transcribe_drums(p, out_mid, bpm=bpm)
    else:
        res = core_transcribe_tonal(p, out_mid, bpm=bpm)
    return str(res)


@mcp.tool()
async def inspect_midi_bars(
    midi_path: str,
    start_bar: int = 1,
    end_bar: int = 8,
    bpm: float = 120.0,
) -> Dict[str, Any]:
    """Read MIDI notes in a bar-by-bar, beat-indexed format optimized for LLM context."""
    p = Path(midi_path).resolve()
    digest = core_generate_midi_digest(p, bpm=bpm, start_bar=start_bar, end_bar=end_bar)
    return digest.model_dump()


@mcp.tool()
async def deconstruct_full_pipeline(file_path: str, six_stems: bool = False) -> Dict[str, Any]:
    """End-to-end pipeline: separate stems, analyze acoustic features, and transcribe to MIDI."""
    p = Path(file_path).resolve()
    ws = WorkspaceManager()

    # 1. Separate stems
    split_res = core_separate_stems(p, workspace=ws, six_stems=six_stems)

    # 2. Analyze
    analysis_src = split_res.stems.get("drums", str(p))
    analysis = core_analyze_audio(analysis_src)
    bpm = analysis.bpm if analysis.bpm > 0 else 120.0
    ws.save_metadata(p, analysis.model_dump())

    # 3. Transcribe stems
    track_dir = ws.get_track_dir(p)
    midis_dir = track_dir / "midis"
    midis_dir.mkdir(parents=True, exist_ok=True)

    midi_map = {}
    for name, stem_p in split_res.stems.items():
        out_mid = midis_dir / f"{name}.mid"
        if name == "drums":
            core_transcribe_drums(stem_p, out_mid, bpm=bpm)
        else:
            core_transcribe_tonal(stem_p, out_mid, bpm=bpm)
        midi_map[name] = str(out_mid)

    return {
        "workspace_dir": str(track_dir),
        "analysis": analysis.model_dump(),
        "stems": split_res.stems,
        "midis": midi_map,
    }


def main():
    """Main entrypoint for MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
