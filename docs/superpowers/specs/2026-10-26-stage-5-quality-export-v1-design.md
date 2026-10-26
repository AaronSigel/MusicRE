# Stage 5: Quality Evaluation, Export & MusicRE v1.0 Packaging Design Document

**Status:** Approved  
**Author:** AI + Human Partner  
**Target:** MusicRE Stage 5 (PR-06: Final v1.0 Release)

## 1. Intent & Purpose

Complete the MusicRE v1.0 release by implementing a two-tier quality evaluation engine (`quality_report.json`), full project export into structured bundles (WAV, MIDI, JSON), execution profiles (`fast`, `balanced`, `quality`), and complete CLI/MCP documentation.

## 2. Requirements & Architecture Decisions

### 2.1 Two-Tier Quality Evaluation (`src/music_re/core/quality.py`)
- `QualityReport` in `src/music_re/schemas.py`:
  - `is_objective: bool` (True if ground truth was provided, False if diagnostic heuristics)
  - `separation_metrics: Dict[str, Any]` (SDR, SIR, SAR in dB when references exist; residual energy / leakage when absent)
  - `transcription_metrics: Dict[str, Any]` (Note Precision, Recall, F-measure when reference MIDI exists; note count and density when absent)
  - `audio_diagnostics: Dict[str, Any]` (peak dBFS, clipping sample count, SNR estimate)
- Never fabricate subjective percentage scores without ground truth.

### 2.2 Project Export Engine (`src/music_re/core/export.py`)
- `export_bundle(file_path: Path | str, target_dir: Path | str, profile: str = "balanced", archive: bool = False) -> Path`:
  - Exports clean artifacts:
    - `stems/`: `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`
    - `midis/`: `vocals.mid`, `drums.mid`, `bass.mid`, `other.mid`, `all_stems.mid`, and `*_raw.mid`
    - `timeline.json` (structure, BPM, beats, downbeats, sections)
    - `manifest.json` (model versions, hashes, stage statuses)
    - `quality_report.json`

### 2.3 Execution Profiles
- `--profile [fast | balanced | quality]`:
  - `fast`: 4-stem Demucs, skips redundant passes, fast quantization.
  - `balanced` (default): full pipeline, timeline alignment, all_stems project.
  - `quality`: full pipeline + comprehensive quality report and raw audit logs.

### 2.4 MCP Server Updates (`src/music_re/mcp_server.py`)
- Expose complete tool suite to LLM agents:
  - `deconstruct(file_path, profile, device)`
  - `get_timeline(file_path)`
  - `evaluate_quality(file_path, ground_truth_dir)`
  - `export_project(file_path, output_dir)`

### 2.5 Documentation & Definition of Done for v1.0
- Update `README.md` with full CLI and MCP usage, supported formats, limitations, and license caveats (CC-BY-NC 4.0 for Demucs weights).
- End-to-end verification proving full pipeline on CPU from input audio to exported bundle.

## 3. Non-Goals
- Cloud hosting / remote web service (MusicRE remains local-first CLI/MCP).
