# Stage 4: MIDI Transcription & Time Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transcribe tonal and drum stems into time-synchronized MIDI parts aligned with `timeline.json`, preserving raw and quantized files, and assembling an integrated multi-track project (`all_stems.mid`).

**Architecture:** Dynamic timeline snapping in `midi_quantize.py`, `TranscriberBackend` interface with `BasicPitchTranscriber` and `DrumTranscriber` adapters, multi-track SMF project builder in `midi_project.py`, and workspace manifest tracking.

**Tech Stack:** Python 3.11, PrettyMIDI, basic-pitch, scipy, numpy, typer.

**Spec:** `docs/superpowers/specs/2026-10-26-stage-4-midi-transcription-design.md`

## Global Constraints

- Python strictly pinned to `>=3.11,<3.12`.
- Zero new machine learning models; Basic Pitch ONNX remains primary tonal transcriber.
- Raw unquantized MIDI must always be preserved as `<stem>_raw.mid`.
- Combined multi-track project must be output as `<workspace>/midis/all_stems.mid`.
- All existing 75 tests must continue to pass without regression.

## Review Focus

- Note timestamps drifting outside audio duration bounds during quantization.
- Hanging or duplicate notes caused by adjacent overlapping intervals of identical pitch.
- Empty or silent stem files producing invalid or zero-track MIDI files.
- Channel conflicts in combined MIDI (drums must strictly route to MIDI channel 10 / is_drum=True).
- Negative note durations when snapped onset passes the original note end.

---

### Task 1: Dynamic Timeline Quantization Engine

**Files:**
- Create: `src/music_re/core/midi_quantize.py`
- Modify: `src/music_re/core/midi_tonal.py`
- Test: `tests/test_midi_quantize.py`

**Interfaces:**
- Produces: `quantize_to_timeline(pm: pretty_midi.PrettyMIDI, timeline: Optional[MusicalTimeline] = None, bpm: float = 120.0, grid_subdivisions: int = 4, min_duration: float = 0.05) -> pretty_midi.PrettyMIDI`

- [ ] **Step 1: Write failing test in `tests/test_midi_quantize.py` validating dynamic timeline beat grid snapping**
- [ ] **Step 2: Run `uv run pytest tests/test_midi_quantize.py` and verify FAIL**
- [ ] **Step 3: Implement `midi_quantize.py` with sub-beat interpolation and pitch-group overlap deduplication**
- [ ] **Step 4: Run `uv run pytest tests/test_midi_quantize.py` and verify PASS**

---

### Task 2: Transcriber Backend Base Interface & Adapters

**Files:**
- Create: `src/music_re/core/backends/transcriber_base.py`
- Create: `src/music_re/core/backends/basic_pitch_backend.py`
- Create: `src/music_re/core/backends/drum_backend.py`
- Modify: `src/music_re/core/backends/__init__.py`
- Test: `tests/test_transcriber_backends.py`

**Interfaces:**
- Produces: `TranscriberBackend` ABC
- Produces: `BasicPitchTranscriber` (tonal mode) and `DrumTranscriber` (drums mode)
- Produces: `get_transcriber_backend(mode: str = "tonal", name: str = "auto") -> TranscriberBackend`

- [ ] **Step 1: Write failing test in `tests/test_transcriber_backends.py` testing transcriber adapters and registry**
- [ ] **Step 2: Run `uv run pytest tests/test_transcriber_backends.py` and verify FAIL**
- [ ] **Step 3: Implement `transcriber_base.py`, `basic_pitch_backend.py`, and `drum_backend.py`**
- [ ] **Step 4: Run `uv run pytest tests/test_transcriber_backends.py` and verify PASS**

---

### Task 3: Multi-Track MIDI Project Aggregator

**Files:**
- Create: `src/music_re/core/midi_project.py`
- Test: `tests/test_midi_project.py`

**Interfaces:**
- Produces: `build_combined_midi_project(stem_midis: Dict[str, Path | str], output_path: Path | str, timeline: Optional[MusicalTimeline] = None, bpm: float = 120.0) -> Path`

- [ ] **Step 1: Write failing test in `tests/test_midi_project.py` testing multi-track SMF assembly and channel assignments**
- [ ] **Step 2: Run `uv run pytest tests/test_midi_project.py` and verify FAIL**
- [ ] **Step 3: Implement `midi_project.py` merging stem tracks with appropriate GM programs and drum channel 10**
- [ ] **Step 4: Run `uv run pytest tests/test_midi_project.py` and verify PASS**

---

### Task 4: Workspace Pipeline & CLI Integration

**Files:**
- Modify: `src/music_re/workspace.py`
- Modify: `src/music_re/cli.py`
- Test: `tests/test_midi_pipeline_integration.py`

**Interfaces:**
- Produces: Integrated `deconstruct` producing `all_stems.mid` and updating `manifest.midis`
- Produces: CLI `music-re to-midi` with `--timeline` flag

- [ ] **Step 1: Write failing test in `tests/test_midi_pipeline_integration.py` testing CLI flags and combined project generation**
- [ ] **Step 2: Run `uv run pytest tests/test_midi_pipeline_integration.py` and verify FAIL**
- [ ] **Step 3: Update `src/music_re/cli.py` to route through transcriber backends, generate `all_stems.mid`, and record in manifest**
- [ ] **Step 4: Run `uv run pytest tests/test_midi_pipeline_integration.py` and verify PASS**

---

### Task 5: End-to-End MIDI Time Synchronization & Regression Suite

**Files:**
- Create: `tests/test_midi_e2e.py`

**Interfaces:**
- Consumes: Complete Stage 4 pipeline
- Produces: Full validation of time-synchronized MIDI tracks across dynamic timeline grids

- [ ] **Step 1: Write comprehensive e2e test in `tests/test_midi_e2e.py` verifying dynamic grid snapping, raw/quantized dual files, and multi-track project**
- [ ] **Step 2: Run `uv run pytest tests/test_midi_e2e.py` and verify PASS**
- [ ] **Step 3: Run full pytest suite across all tests (75+ tests) and verify 100% green**
