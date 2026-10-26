# MusicRE (Audio Reverse Engineering Engine)

MusicRE is a local-first Python application, CLI, and MCP server for music decomposition and reverse engineering. It transforms local MP3, WAV, FLAC, and M4A audio files into separated stems, aligned MIDI parts, musical structure timelines, and diagnostic quality reports.

---

## Architecture & Principles

- **Zero-Reinvention:** Leverages best-in-class upstream ML libraries (Demucs, Basic Pitch, Librosa, PrettyMIDI) via modular adapter interfaces without custom neural training.
- **API-First & Modular:** S換eable engine interfaces (`SeparatorBackend`, `StructureAnalyzerBackend`, `TranscriberBackend`) with runtime device selection (`cpu`, `cuda`, `auto`).
- **Deterministic & Resumable:** Workspaces indexed by SHA-256 streaming content hashes. Atomic writes and `manifest.json` state tracking prevent corrupted caches and redundant processing.
- **Dynamic Time Alignment:** Note events and drum onsets snap to dynamic beat and bar grids (`timeline.json`), preventing tempo drift across long tracks.
- **Honest Evaluation:** Two-tier evaluation (`quality_report.json`) separates objective metrics (SDR/SIR/SAR and note onset F1 when ground truth is provided) from non-fabricated diagnostic heuristics.

---

## Core Pipeline & Artifacts

1. **Audio Ingestion & Normalization:** Unified FFmpeg decoding to 32-bit float PCM with peak normalization to -1.0 dBFS and sample-exact duration preservation.
2. **Stem Separation:** 4-stem (`htdemucs`: vocals, drums, bass, other) and 6-stem (`htdemucs_6s`: + guitar, piano) with memory management (`unload()` cache clearing).
3. **Structure Analysis:** BPM, beat timestamps, downbeats (bars), and contiguous section boundaries (intro, verse, chorus, bridge, outro) saved to `timeline.json`.
4. **MIDI Transcription:**
   - Melodic stems: Basic Pitch ONNX runtime polyphonic transcription with pitch-group overlap deduplication.
   - Percussion: Multi-band frequency energy detection (Kick MIDI 36, Snare MIDI 38, Hi-Hat MIDI 42).
   - Dual export: Unquantized raw (`<stem>_raw.mid`) and timeline-quantized (`<stem>.mid`).
   - Integrated project: Single multi-track SMF Type 1 project (`all_stems.mid`) with General MIDI programs and Drum Channel 10.
5. **Quality Report & Export:** Generates `quality_report.json` and exports clean bundles (folder or `.zip`).

---

## Installation

Requires **Python 3.11** and **FFmpeg** installed in your system PATH.

```bash
# Clone and install using uv (recommended)
git clone https://github.com/AaronSigel/MusicRE.git
cd MusicRE
uv sync
```

---

## CLI Usage

### Full End-to-End Deconstruction
```bash
# Run full reverse engineering pipeline (split -> analyze -> structure -> midi -> evaluate)
uv run music-re deconstruct input.mp3

# Use GPU acceleration and 6-stem model
uv run music-re deconstruct input.flac --six-stems --device cuda --profile quality
```

### Individual Commands
```bash
# Extract an audio segment
uv run music-re slice song.wav --start 15.0 --end 30.0

# Acoustic analysis (BPM, Key, LUFS, Beat Grid)
uv run music-re analyze song.wav [--json]

# Separate stems using Demucs adapter
uv run music-re split song.wav [--six-stems] [--device auto|cpu|cuda]

# Extract musical structure & sections (timeline.json)
uv run music-re structure song.wav [--backend auto|librosa|allin1] [--json]

# Transcribe isolated stem to MIDI with timeline synchronization
uv run music-re to-midi vocals.wav --mode tonal --timeline timeline.json --grid 4
uv run music-re to-midi drums.wav --mode drums --timeline timeline.json

# Inspect MIDI notes by bars for LLM reasoning
uv run music-re inspect-midi midis/vocals.mid --bars 1-8 --bpm 120.0

# Two-tier quality evaluation
uv run music-re evaluate song.wav [--gt-stems path/to/reference_stems] [--json]

# Export clean deliverable bundle or ZIP
uv run music-re export song.wav --out ./output_bundle [--zip] [--profile balanced]
```

---

## MCP Server (Model Context Protocol)

MusicRE includes a native MCP server for Claude Desktop, OpenCode, and AI coding agents.

### Start MCP Server
```bash
uv run music-re-mcp
```

### Available Tools
- `slice_audio`: Cut audio segments to save compute time.
- `analyze_track`: Return BPM, Key, LUFS, and beat timestamps.
- `separate_stems`: Separate track into isolated WAV stems.
- `get_structure_timeline`: Extract sections, beats, and downbeats into `timeline.json`.
- `transcribe_stem`: Convert stem to MIDI with timeline synchronization.
- `inspect_midi_bars`: Bar-by-bar, beat-indexed note representation for reasoning.
- `evaluate_quality_report`: Produce two-tier quality evaluation report.
- `export_project_bundle`: Package deliverables into structured folders or ZIP archives.
- `deconstruct_full_pipeline`: Execute complete reverse engineering workflow.

### Claude Desktop Configuration (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "music-re": {
      "command": "uv",
      "args": ["--directory", "/path/to/MusicRE", "run", "music-re-mcp"]
    }
  }
}
```

---

## Licenses & Pretrained Weights

- **Code:** Licensed under the [MIT License](LICENSE).
- **Basic Pitch:** Apache 2.0 (code and ONNX weights open for commercial use).
- **Demucs Pretrained Weights (`htdemucs` / `htdemucs_6s`):** Pretrained on Musdb-HQ under **CC-BY-NC 4.0** (non-commercial research/personal use only).
- **Candidate Models:** Check dataset licenses prior to commercial deployment.
