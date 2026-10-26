# Stage 3: Musical Structure Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract BPM, beats, downbeats (measure starts), and musical sections (intro, verse, chorus, outro) into a standardized `timeline.json` using pre-separated stems with zero duplicate separation.

**Architecture:** `StructureAnalyzerBackend` ABC, `LibrosaStructureAnalyzer` baseline, `AllInOneStructureAnalyzer` adapter reusing Demucs stems, registry auto-selection factory, and `timeline.json` persistence in `WorkspaceManager`.

**Tech Stack:** Python 3.11, Librosa, scipy, numpy, all-in-one-infer (optional neural candidate), pydantic.

**Spec:** `docs/superpowers/specs/2026-10-26-stage-3-musical-structure-design.md`

## Global Constraints

- Python strictly pinned to `>=3.11,<3.12`.
- Pre-separated Demucs stems from Stage 2 must be reused; do not run Demucs separation twice.
- Output artifact must be named `timeline.json` inside the track workspace directory.
- All existing 62 tests must continue to pass without regression.

## Review Focus

- Beat and downbeat mismatch (downbeats not aligning with beat indices).
- Section boundaries extending beyond track duration or having negative durations.
- Missing stems causing unhandled crashes instead of falling back to mix audio.
- Non-contiguous or overlapping section partitions.
- Empty or zero-length audio inputs causing division by zero in segmentation kernels.

---

### Task 1: Musical Timeline Schemas

**Files:**
- Modify: `src/music_re/schemas.py`
- Test: `tests/test_timeline_schemas.py`

**Interfaces:**
- Produces: `TimelineSection(label, start_sec, end_sec, confidence)`
- Produces: `MusicalTimeline(duration_sec, bpm, beats, downbeats, sections)`
- Produces: `WorkspaceManifest.timeline: Optional[Dict[str, Any]] = None`

- [ ] **Step 1: Write failing test in `tests/test_timeline_schemas.py` validating timeline schemas**
- [ ] **Step 2: Run `uv run pytest tests/test_timeline_schemas.py` and verify FAIL**
- [ ] **Step 3: Implement `TimelineSection` and `MusicalTimeline` in `src/music_re/schemas.py`**
- [ ] **Step 4: Run `uv run pytest tests/test_timeline_schemas.py` and verify PASS**

---

### Task 2: Structure Analyzer Base Interface & Librosa Fallback

**Files:**
- Create: `src/music_re/core/backends/structure_base.py`
- Create: `src/music_re/core/backends/librosa_structure_backend.py`
- Test: `tests/test_librosa_structure.py`

**Interfaces:**
- Produces: `StructureAnalyzerBackend` ABC with `analyze(audio_path, stems=None, device="auto") -> MusicalTimeline`
- Produces: `LibrosaStructureAnalyzer(StructureAnalyzerBackend)`

- [ ] **Step 1: Write failing test in `tests/test_librosa_structure.py` for beat, downbeat, and section extraction**
- [ ] **Step 2: Run `uv run pytest tests/test_librosa_structure.py` and verify FAIL**
- [ ] **Step 3: Implement `structure_base.py` and `librosa_structure_backend.py` using spectral novelty and recurrence segmentation**
- [ ] **Step 4: Run `uv run pytest tests/test_librosa_structure.py` and verify PASS**

---

### Task 3: All-In-One-Infer Adapter & Registry Auto-Selection

**Files:**
- Create: `src/music_re/core/backends/allin1_backend.py`
- Modify: `src/music_re/core/backends/__init__.py`
- Test: `tests/test_structure_registry.py`

**Interfaces:**
- Produces: `AllInOneStructureAnalyzer(StructureAnalyzerBackend)`
- Produces: `get_structure_analyzer(name="auto", device="auto") -> StructureAnalyzerBackend`

- [ ] **Step 1: Write failing test in `tests/test_structure_registry.py` testing registry lookup and fallback behavior**
- [ ] **Step 2: Run `uv run pytest tests/test_structure_registry.py` and verify FAIL**
- [ ] **Step 3: Implement `allin1_backend.py` (with stem directory reuse) and register in `backends/__init__.py`**
- [ ] **Step 4: Run `uv run pytest tests/test_structure_registry.py` and verify PASS**

---

### Task 4: Workspace Pipeline & CLI Integration

**Files:**
- Modify: `src/music_re/workspace.py`
- Modify: `src/music_re/cli.py`
- Test: `tests/test_structure_cli.py`

**Interfaces:**
- Produces: `WorkspaceManager.save_timeline(file_path, timeline)` and `load_timeline(file_path)`
- Produces: CLI `music-re structure <file>` and updated `music-re deconstruct`

- [ ] **Step 1: Write failing test in `tests/test_structure_cli.py` testing timeline save/load and CLI structure command**
- [ ] **Step 2: Run `uv run pytest tests/test_structure_cli.py` and verify FAIL**
- [ ] **Step 3: Implement timeline methods in `workspace.py`, wire CLI command `structure` and integrate into `cmd_deconstruct`**
- [ ] **Step 4: Run `uv run pytest tests/test_structure_cli.py` and verify PASS**

---

### Task 5: End-to-End Musical Structure Regression Suite

**Files:**
- Create: `tests/test_structure_e2e.py`

**Interfaces:**
- Consumes: Complete Stage 3 pipeline
- Produces: Verification of full timeline generation on reference audio without re-running separation

- [ ] **Step 1: Write comprehensive e2e test in `tests/test_structure_e2e.py` verifying `timeline.json` structure, beats coherence, and stem reuse**
- [ ] **Step 2: Run `uv run pytest tests/test_structure_e2e.py` and verify PASS**
- [ ] **Step 3: Run full pytest suite across all tests (62+ tests) and verify 100% green**
