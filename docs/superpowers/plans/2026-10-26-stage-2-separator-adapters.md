# Stage 2: Modular Separator Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create extensible `SeparatorBackend` interface, migrate Demucs into a dedicated adapter with 4-stem and 6-stem support, model reuse, and device management.

**Architecture:** Abstract base class `SeparatorBackend` with registry factory in `src/music_re/core/backends/`, concrete `DemucsSeparator` adapter with automatic device selection and safe unload, refactored `separator.py` delegating to the backend registry.

**Tech Stack:** Python 3.11, PyTorch 2.2+, Demucs 4.1.0, soundfile, typer.

**Spec:** `docs/superpowers/specs/2026-10-26-stage-2-separator-adapters-design.md`

## Global Constraints

- Python strictly pinned to `>=3.11,<3.12`.
- Zero new machine learning models; Demucs remains primary backend.
- Full backwards compatibility for `separate_stems(...)` signature and CLI.
- All existing 50 tests must continue to pass without regression.

## Review Focus

- Memory leaks from persistent model instances when `unload()` is called.
- Incompatible device strings crashing PyTorch tensors (e.g. invalid CUDA index when running on CPU).
- Cache path divergence between legacy `separate_stems` and `DemucsSeparator`.
- 6-stem mode bleeding/quality warnings omitted from output metadata.
- Corrupted stem files written if separation is interrupted midway.

---

### Task 1: SeparatorBackend Abstract Interface & Registry

**Files:**
- Create: `src/music_re/core/backends/separator_base.py`
- Create: `src/music_re/core/backends/__init__.py`
- Test: `tests/test_separator_base.py`

**Interfaces:**
- Produces: `SeparatorBackend` ABC with `name`, `model_name`, `supported_stems`, `device`, `load()`, `unload()`, `separate()`
- Produces: `get_separator_backend(name: str, ...)` and `register_separator_backend(name: str, cls)`

- [ ] **Step 1: Write failing test in `tests/test_separator_base.py` for interface methods and registry**
- [ ] **Step 2: Run `uv run pytest tests/test_separator_base.py` and verify FAIL**
- [ ] **Step 3: Implement `separator_base.py` and registry factory in `backends/__init__.py`**
- [ ] **Step 4: Run `uv run pytest tests/test_separator_base.py` and verify PASS**

---

### Task 2: Demucs Backend Adapter Implementation

**Files:**
- Create: `src/music_re/core/backends/demucs_backend.py`
- Modify: `src/music_re/core/backends/__init__.py`
- Test: `tests/test_demucs_backend.py`

**Interfaces:**
- Produces: `DemucsSeparator(SeparatorBackend)` registered as `"demucs"`
- Produces: Device resolution (`"auto"`, `"cpu"`, `"cuda"`, `"rocm"`) and memory management

- [ ] **Step 1: Write failing test in `tests/test_demucs_backend.py` for model instantiation, device detection, 4/6 stems, and unload**
- [ ] **Step 2: Run `uv run pytest tests/test_demucs_backend.py` and verify FAIL**
- [ ] **Step 3: Implement `DemucsSeparator` wrapping `htdemucs` and `htdemucs_6s` with atomic stem writing**
- [ ] **Step 4: Run `uv run pytest tests/test_demucs_backend.py` and verify PASS**

---

### Task 3: Refactor Core Separator & CLI Integration

**Files:**
- Modify: `src/music_re/core/separator.py`
- Modify: `src/music_re/cli.py`
- Test: `tests/test_separator.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `DemucsSeparator` from `src/music_re/core/backends`
- Produces: Backwards-compatible `separate_stems(file_path, workspace, six_stems, device, progress_callback)`
- Produces: CLI `--device` option in `split` and `deconstruct` commands

- [ ] **Step 1: Update existing tests in `tests/test_separator.py` and `tests/test_cli.py` for `--device` flag**
- [ ] **Step 2: Run tests to verify test expectations**
- [ ] **Step 3: Refactor `src/music_re/core/separator.py` to use `DemucsSeparator` and wire `--device` in `src/music_re/cli.py`**
- [ ] **Step 4: Run `uv run pytest tests/test_separator.py tests/test_cli.py` and verify PASS**

---

### Task 4: Extensibility, Custom Backend & End-to-End Regression Tests

**Files:**
- Create: `tests/test_separator_extensibility.py`

**Interfaces:**
- Consumes: Entire backend adapter framework
- Produces: Verification of third-party pluggable separator, model reuse, and full regression pass

- [ ] **Step 1: Write tests in `tests/test_separator_extensibility.py` registering a mock separator backend and verifying pipeline execution**
- [ ] **Step 2: Run `uv run pytest tests/test_separator_extensibility.py` and verify PASS**
- [ ] **Step 3: Run full pytest suite across all tests (50+ tests) and verify 100% green**
