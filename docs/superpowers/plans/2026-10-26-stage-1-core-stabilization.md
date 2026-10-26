# Stage 1: Core Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize MusicRE core audio decoding, caching, analysis, and MIDI processing with zero new ML models.

**Architecture:** Replace audioread with a robust FFmpeg temp WAV decoder, isolate separation model caches with SHA-256 content IDs, remove artificial beat limits, eliminate key detection on drums, preserve raw and quantized MIDI, and add resumable execution status to the workspace.

**Tech Stack:** Python 3.11, FFmpeg (via subprocess), soundfile, librosa, pretty_midi, demucs, basic-pitch, pydantic.

**Spec:** `docs/superpowers/specs/2026-10-26-stage-1-core-stabilization-design.md`

## Global Constraints

- Python strictly pinned to `>=3.11,<3.12`.
- Zero new machine learning models; use existing installed packages.
- All temporary files must be cleaned up via context managers (`tempfile.TemporaryDirectory`).
- All existing 33 tests must continue to pass without regression.

## Review Focus

- FFmpeg subprocess failures on corrupted audio or missing FFmpeg binary.
- Audio sample-rate drift or duration mismatch during decoding.
- 4-stem and 6-stem cache collisions corrupting subsequent separations.
- Infinite loops or index errors on tracks with >1000 beats.
- Overlapping MIDI notes of identical pitch causing hanging notes or note dropouts in DAW.

---

### Task 1: Unified FFmpeg Audio Decoder & Normalization

**Files:**
- Create: `src/music_re/core/audio_io.py`
- Modify: `src/music_re/core/analyzer.py`
- Test: `tests/test_audio_io.py`

**Interfaces:**
- Produces: `decode_audio(input_path: Path | str, target_sr: int | None = None, normalize_db: float | None = -1.0) -> Tuple[np.ndarray, int]`
- Produces: `decode_to_temp_wav(input_path: Path | str, target_sr: int | None = None) -> Generator[Path, None, None]`

- [ ] **Step 1: Write failing test for `decode_audio` and `decode_to_temp_wav` in `tests/test_audio_io.py`**
- [ ] **Step 2: Run `uv run pytest tests/test_audio_io.py` and verify FAIL**
- [ ] **Step 3: Implement `audio_io.py` using `ffmpeg -i ... -c:a pcm_f32le ...` and `soundfile`**
- [ ] **Step 4: Update `src/music_re/core/analyzer.py` to use `audio_io.decode_audio` instead of direct `librosa.load(audioread)`**
- [ ] **Step 5: Run `uv run pytest tests/test_audio_io.py tests/test_analyzer.py` and verify PASS without audioread deprecation warnings**

---

### Task 2: Content-Based Hashing & Model Cache Segregation

**Files:**
- Modify: `src/music_re/workspace.py`
- Modify: `src/music_re/core/separator.py`
- Test: `tests/test_workspace_cache.py`

**Interfaces:**
- Consumes: Audio file path
- Produces: `compute_file_content_hash(path: Path | str) -> str`
- Produces: Segregated cache paths: `<cache_dir>/<model_name>/<content_hash>/`

- [ ] **Step 1: Write failing test in `tests/test_workspace_cache.py` verifying 4-stem and 6-stem separation cache separation and content hashing**
- [ ] **Step 2: Run `uv run pytest tests/test_workspace_cache.py` and verify FAIL**
- [ ] **Step 3: Implement SHA-256 content hashing in `workspace.py` and model-specific subdirectories in `separator.py`**
- [ ] **Step 4: Run `uv run pytest tests/test_workspace_cache.py tests/test_separator.py` and verify PASS**

---

### Task 3: Remove 64-Beat Limit & Fix Drum Key Detection

**Files:**
- Modify: `src/music_re/core/analyzer.py`
- Modify: `src/music_re/core/slicer.py`
- Modify: `src/music_re/schemas.py`
- Test: `tests/test_analyzer_unlimited.py`

**Interfaces:**
- Produces: Full duration beats array in `AudioAnalysisResult`
- Produces: Key estimation that ignores drum stems

- [ ] **Step 1: Write failing test in `tests/test_analyzer_unlimited.py` for >64 beats and drum key skipping**
- [ ] **Step 2: Run `uv run pytest tests/test_analyzer_unlimited.py` and verify FAIL**
- [ ] **Step 3: Remove `[:64]` slice limits from `analyzer.py` and `slicer.py`; guard key detection against 'drums' stem**
- [ ] **Step 4: Run `uv run pytest tests/test_analyzer_unlimited.py tests/test_analyzer.py` and verify PASS**

---

### Task 4: MIDI Deduplication, Overlap Handling & Raw Preservation

**Files:**
- Modify: `src/music_re/core/midi_tonal.py`
- Modify: `src/music_re/schemas.py`
- Test: `tests/test_midi_cleaning.py`

**Interfaces:**
- Produces: `clean_and_quantize_midi(input_midi, output_midi, ...)` with overlap resolution
- Produces: Both `stem_raw.mid` and `stem_quantized.mid` (and default `stem.mid`)

- [ ] **Step 1: Write failing test in `tests/test_midi_cleaning.py` with overlapping identical pitch notes and dual MIDI output check**
- [ ] **Step 2: Run `uv run pytest tests/test_midi_cleaning.py` and verify FAIL**
- [ ] **Step 3: Enhance `clean_and_quantize_midi` to truncate overlapping same-pitch notes and emit both raw and quantized MIDI files**
- [ ] **Step 4: Run `uv run pytest tests/test_midi_cleaning.py tests/test_midi_tonal.py` and verify PASS**

---

### Task 5: Resumable Pipeline & Stage Diagnostic Error Reporting

**Files:**
- Modify: `src/music_re/workspace.py`
- Modify: `src/music_re/schemas.py`
- Modify: `src/music_re/cli.py`
- Test: `tests/test_pipeline_resumption.py`

**Interfaces:**
- Produces: `WorkspaceManifest` with `status`, `stage`, `stage_errors`
- Produces: Safe resume mechanism in CLI and workspace

- [ ] **Step 1: Write failing test in `tests/test_pipeline_resumption.py` for error recording and resuming skipped completed stages**
- [ ] **Step 2: Run `uv run pytest tests/test_pipeline_resumption.py` and verify FAIL**
- [ ] **Step 3: Update `schemas.py`, `workspace.py`, and `cli.py` to record stage transitions, catch errors into manifest, and skip completed stages**
- [ ] **Step 4: Run `uv run pytest tests/test_pipeline_resumption.py tests/test_cli.py` and verify PASS**

---

### Task 6: End-to-End Core Stabilization Integration Suite

**Files:**
- Create: `tests/test_stage1_e2e.py`

**Interfaces:**
- Consumes: Entire stabilized pipeline
- Produces: Verification of full workflow on MP3, WAV, and FLAC inputs

- [ ] **Step 1: Write end-to-end test in `tests/test_stage1_e2e.py` covering multi-format decode, cache isolation, unlimited beats, dual MIDI, and manifest status**
- [ ] **Step 2: Run `uv run pytest tests/test_stage1_e2e.py` and verify PASS**
- [ ] **Step 3: Run full pytest suite across all tests and verify 100% green without deprecation warnings**
