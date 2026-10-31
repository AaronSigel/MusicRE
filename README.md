# MusicRE (Audio Reverse Engineering Engine)

MusicRE is a local-first Python application, CLI, and MCP server for music decomposition and reverse engineering. It transforms local MP3, WAV, FLAC, and M4A audio files into separated stems, aligned MIDI parts, musical structure timelines, and diagnostic quality reports.

---

## Architecture & Principles

- **Zero-Reinvention:** Leverages best-in-class upstream ML libraries (Demucs, Basic Pitch, Librosa, PrettyMIDI) via modular adapter interfaces without custom neural training.
- **API-First & Modular:** Swappable engine interfaces (`SeparatorBackend`, `StructureAnalyzerBackend`, `TranscriberBackend`) with runtime device selection (`cpu`, `cuda`, `auto`). CLI and MCP use the same pipeline service.
- **Deterministic & Resumable:** Workspaces indexed by SHA-256 streaming content hashes. Atomic writes and `manifest.json` state tracking prevent corrupted caches and redundant processing.
- **Dynamic Time Alignment:** Note events and drum onsets snap to dynamic beat and bar grids (`timeline.json`), preventing tempo drift across long tracks.
- **Honest Evaluation:** `museval` BSSEval v4 SDR/ISR/SIR/SAR and `mir_eval` pitch+onset/offset scores require actual reference comparisons. Each metric group reports its mode and coverage. Missing comparisons and undefined scores are not successes.

---

## Core Pipeline & Artifacts

1. **Audio Ingestion & Normalization:** Unified FFmpeg decoding to 32-bit float PCM with peak normalization to -1.0 dBFS and sample-exact duration preservation.
2. **Stem Separation:** 4-stem (`htdemucs`: vocals, drums, bass, other) and 6-stem (`htdemucs_6s`: + guitar, piano) with memory management (`unload()` cache clearing).
3. **Structure Analysis:** All-In-One-Infer (`allin1_infer`, pinned in the lock) supplies neural beats, downbeats, and section labels while reusing the existing stems. Librosa is a baseline: labels are `unknown`, confidence is `null`, and downbeats are unavailable. Unreported neural confidence is also `null`.
4. **MIDI Transcription:**
   - Melodic stems: Basic Pitch ONNX runtime polyphonic transcription with pitch-group overlap deduplication.
   - Percussion: Multi-band frequency energy detection (Kick MIDI 36, Snare MIDI 38, Hi-Hat MIDI 42).
   - Dual export: Unquantized raw (`<stem>_raw.mid`) and timeline-quantized (`<stem>.mid`).
   - Integrated project: Single multi-track SMF Type 1 project (`all_stems.mid`) with General MIDI programs and Drum Channel 10.
5. **Quality Report & Export:** Generates `quality_report.json` and exports clean bundles (folder, `.zip`, or `.dawproject`).
   - **DAWproject Export (`--dawproject` / `-d`):** Packages stems, aligned MIDI notes, tempo map, and section markers into a standard Bitwig/Studio One project.

---

## Installation

Requires **Python 3.11** and **FFmpeg** installed in your system PATH.

```bash
# Clone and install using uv (recommended)
git clone https://github.com/AaronSigel/MusicRE.git
cd MusicRE
uv sync --locked --extra dev
uv run --locked pytest -m 'not models and not real_e2e' -q
```

---

## CLI Usage

### Full End-to-End Deconstruction
```bash
# Run full reverse engineering pipeline (split -> analyze -> structure -> midi -> evaluate)
uv run --locked music-re deconstruct input.mp3 --cache-root ./data/cache

# Use GPU acceleration and 6-stem model
uv run music-re deconstruct input.flac --six-stems --device cuda --profile quality
```

### 4. Transcribe stem to MIDI or inspect
```bash
# Transcribe vocals with continuous CREPE pitch-bends
uv run music-re to-midi vocals.wav --mode tonal --timeline timeline.json

# Transcribe drums via deep-learning ADTOF (5 classes)
uv run music-re to-midi drums.wav --mode drums --timeline timeline.json

# Inspect notes across bar ranges (all, tonal, or drums)
uv run music-re inspect-midi midis/drums.mid --bars 1-8 --bpm 120.0 --part drums
```

### 5. Export DAWProject for Bitwig / Studio One / Reaper
```bash
# Export single DAWProject bundle with stems, MIDI, tempo map & markers
uv run music-re export song.wav --out song.dawproject --dawproject
```

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

## Execution profiles and result states

| Setting | fast | balanced | quality |
| --- | --- | --- | --- |
| Stems | 4 | requested 4/6 | requested 4/6 |
| Demucs shifts / overlap | 0 / 0.1 | 1 / 0.25 | 2 / 0.5 |
| Structure | Librosa baseline | Harmonix fold 0 | Harmonix 8-fold ensemble |
| Raw MIDI / diagnostics | no / no | yes / yes | yes / yes |
| Grid / note endings | 4 subdivisions / snapped | 4 / snapped | 4 / snapped |

These are compute policies, not measured quality guarantees. Neural inference failure is an error, not a silent baseline fallback. `structure --backend auto` only falls back when the neural package itself is absent. The standard locked install includes it.

Transcription results distinguish `notes` and successful `no_notes`; missing/corrupt model output raises an error. MCP `transcribe_stem` returns structured result metadata instead of a path string. The manifest records settings, packages, model IDs, available checkpoint SHA-256 values, stage fingerprints, checksums, and per-stem result states. Changing settings invalidates dependent stages; old unversioned runs are not trusted for resumption.

Export requires a completed, valid active artifact set. Its profile must match the run profile. It refuses existing destinations and partial/corrupt workspaces. Bundled artifact paths are relative; original input location remains provenance. Stale raw MIDI from earlier runs is not included.

Diagnostics are not accuracy scores. `rms_to_p10_amplitude_db` is an amplitude-distribution statistic, **not SNR**. Undefined dB values are `null`. Reference separation uses fixed source order, whole-track BSSEval v4 frames at 22,050 Hz, no source permutation, and rejects mismatched shapes. Pitched transcription uses 50 ms onset tolerance, 50 cents pitch tolerance, and optional offsets within 20% or 50 ms. GM drums are scored separately by onset and pitch, without offsets.

## Spectral MIDI validation

The optional `spectral_validation` stage runs after transcription and checks each transcribed note against the stem audio (STFT/CQT harmonic support, spectral contrast, temporal persistence, onset agreement, leakage). It uses NumPy, SciPy and Librosa only: no extra models, no GPU.

```bash
uv run --locked music-re deconstruct song.wav --profile balanced --spectral-validation audit
```

MCP `deconstruct_full_pipeline` takes the same `spectral_validation` argument. Values are case-insensitive.

| Mode | What it does |
| --- | --- |
| `off` (default) | Stage does nothing. Pipeline output is unchanged. |
| `audit` | Computes per-note evidence and writes `spectral_validation.json`. MIDI files stay byte-identical. Every note is `unscored`. |
| `filter` | **Experimental.** Also writes filtered MIDI to `midis_validated/` (per stem plus `all_stems.mid`). Originals in `midis/` are never touched. Evaluation, export and DAWproject use the filtered set. |

Filter rules: a note is rejected only when both harmonic support and temporal persistence fall below per-role thresholds; uncertain notes are kept. Only `bass`, `guitar` and `other` are filtered. `piano` and `vocals` stay audit-only, `drums` are never filtered, and unknown stems pass through with a warning. The thresholds are **not yet calibrated** against benchmark data, and `note_score` stays `null` until they are. Do not treat `filter` as an accuracy improvement until the Slakh A/B (`scripts/run_quality_benchmarks.py --spectral-validation off audit filter --split test`) shows one.

Workspace artifacts: `spectral_validation.json` holds the full per-note report. The manifest keeps only a summary (mode, algorithm version, active MIDI variant, report path, per-stem status/counts/warnings). In filter mode an exported bundle puts the filtered MIDI in `midis/` and the originals in `midis_original/`. Changing the mode or validator settings reruns only `spectral_validation` and `evaluation`; separation, analysis, structure and transcription are reused from cache.

Related change: chord detection now computes one whole-track chroma pass and slices it per bar, instead of a CQT per bar. Chord labels may differ slightly at bar boundaries.

Known limit: feature extraction is chunked, but the retained per-stem magnitude spectrogram grows linearly with duration, about 42 MiB per minute per stem. A 10-minute track can add several hundred MiB of peak RAM.

## Verification

CI runs the locked CPU suite without downloaded-weight tests. Run those explicitly with `uv run --locked pytest -m models -q`. Real-model acceptance requires FFmpeg, model downloads, an MP3 whose use is permitted, and separate CLI/MCP caches; it is not established by synthetic or mocked tests. See `docs/verification/` for executed evidence and limitations.

## Benchmarking against Slakh2100

`scripts/fetch_slakh_subset.py` fetches a few tracks from BabySlakh (a 20-track subset of Slakh2100), `scripts/slakh_mapping.py` maps each track's per-stem MIDI and audio into MusicRE's four groups (`vocals`/`drums`/`bass`/`other`) and builds an aligned ground truth, and `scripts/run_quality_benchmarks.py` runs the real pipeline against that ground truth plus any `--real-track` files you point it at with no ground truth (manual review only), then writes `BenchmarkReport.json`/`BenchmarkReport.md` and a timestamped snapshot under `data/benchmarks/history/` for comparing later versions:

```bash
uv run python scripts/run_quality_benchmarks.py --real-track /path/to/your/own/track.mp3
```

The real BabySlakh download (882.8 MB) is opt-in: `MUSIC_RE_TEST_SLAKH=1 uv run --locked pytest -m dataset -q`. BabySlakh is CC-BY-4.0 (Manilow, Wichern, Seetharaman, Le Roux, *Cutting Music Source Separation Some Slakh*, WASPAA 2019); this repo fetches it on demand and never commits its audio. Known gap: Slakh has no sung-vocal stems, so the `vocals` group has no objective score from this dataset — only the real-track spot checks can speak to vocals.

## Licenses & Pretrained Weights

- **Code:** Licensed under [Apache 2.0](LICENSE).
- **Basic Pitch:** Apache 2.0 (code and ONNX weights open for commercial use).
- **Demucs:** [Upstream states MIT](https://github.com/facebookresearch/demucs#license). Exact checkpoint commercial-use rights are not independently verified here. Do not infer a checkpoint license solely from a training dataset license.
- **All-In-One-Infer:** [MIT code](https://github.com/openmirlab/all-in-one-infer); review exact Harmonix checkpoints and transitive dependencies separately.
- Input recordings, dataset rights, code licenses, and weight licenses are separate checks. `scripts/audit_licenses.py` records metadata and unresolved items, not legal clearance.
