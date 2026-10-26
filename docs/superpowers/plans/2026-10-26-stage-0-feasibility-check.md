# Stage 0: Feasibility & Environment Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify existing MusicRE code, dependency compatibility, hardware acceleration (CPU/CUDA/ROCm), licenses, and produce `FeasibilityReport.md` with baseline benchmarks.

**Architecture:** Automated verification suite and inspection scripts checking core models (Demucs, Basic Pitch), testing candidate library `all-in-one-infer`, analyzing license compliance, and measuring execution on reference audio.

**Tech Stack:** Python 3.11, uv, PyTorch 2.2+, Demucs 4.0.1, Basic Pitch 0.3.3, Librosa 0.10.2, PrettyMIDI 0.2.10, all-in-one-infer (candidate).

**Spec:** `MusicRE — Implementation Roadmap: Этап 0 (PR-01)`

## Global Constraints

- Python version strictly pinned to `>=3.11,<3.12` (Python 3.11 base environment).
- Zero-reinvention: use upstream APIs, no custom model architectures.
- Unverified components must NOT be added to mandatory dependencies.
- Deliverable artifact: `FeasibilityReport.md` at root.

## Review Focus

- Dependency conflict between `all-in-one-infer`, `demucs`, `basic-pitch`, and `torch/numpy`.
- Model weight download failures or headless/offline execution traps.
- Incompatible licenses in model weights vs MIT code license.
- Non-working GPU acceleration falling back silently without diagnostics.
- Audio drift or sample rate mismatch during decoding on reference files.

---

### Task 1: Lock Python 3.11 & Audit Existing Test Suite

**Files:**
- Create: `.python-version`
- Modify: `pyproject.toml:6`
- Verify: `tests/`

**Interfaces:**
- Consumes: Existing pytest test suite
- Produces: Pinned Python 3.11 environment and clean baseline test run

- [x] **Step 1: Write `.python-version` with `3.11`**
- [x] **Step 2: Pin `requires-python = ">=3.11,<3.12"` in `pyproject.toml`**
- [x] **Step 3: Run `uv run pytest` to confirm all 25 baseline tests pass**
- [x] **Step 4: Record deprecation warnings (audioread, pkg_resources) for remediation in Stage 1**

---

### Task 2: Dependency & License Compatibility Audit (Including All-In-One-Infer)

**Files:**
- Create: `scripts/audit_licenses.py`
- Test: `scripts/audit_licenses.py`

**Interfaces:**
- Consumes: `uv.lock`, PyPI metadata
- Produces: License report table for dependencies, models, and pretrained weights

- [x] **Step 1: Write script `scripts/audit_licenses.py` to inspect installed package licenses and weight licenses**
- [x] **Step 2: Test candidate library `all-in-one-infer` resolution in an isolated dry-run (`uv pip install --dry-run all-in-one-infer`)**
- [x] **Step 3: Document license compatibility (MIT, Apache 2.0, LGPL, non-commercial weight clauses) in output log**

---

### Task 3: Hardware Acceleration & Model Loading Verification

**Files:**
- Create: `scripts/verify_hardware_and_models.py`

**Interfaces:**
- Consumes: `torch`, `demucs`, `basic_pitch`
- Produces: Hardware capability report (CPU, CUDA, ROCm) and weight download/cache verification

- [x] **Step 1: Write verification script `scripts/verify_hardware_and_models.py`**
- [x] **Step 2: Check device detection: `torch.cuda.is_available()`, device name, ROCm HIP flags**
- [x] **Step 3: Test headless CPU execution and weight loading for Demucs (`htdemucs`) and Basic Pitch ONNX**
- [x] **Step 4: Run script via `uv run python scripts/verify_hardware_and_models.py` and verify zero-crash execution**

---

### Task 4: Reference Audio & Benchmark Pipeline Setup

**Files:**
- Create: `scripts/generate_reference_audio.py`
- Create: `scripts/run_benchmarks.py`

**Interfaces:**
- Consumes: Reference audio signals (multitrack synthetic WAV with tone + drums + bass)
- Produces: Timing and memory metrics for Demucs separation, Librosa analysis, Basic Pitch transcription

- [x] **Step 1: Write `scripts/generate_reference_audio.py` to produce standardized 10s multi-stem test WAV (mono & stereo)**
- [x] **Step 2: Write `scripts/run_benchmarks.py` measuring latency, RAM, and output shape across CPU and GPU (if present)**
- [x] **Step 3: Run benchmark on reference audio: `uv run python scripts/run_benchmarks.py`**

---

### Task 5: Compile FeasibilityReport.md & Roadmap Acceptance Gate

**Files:**
- Create: `FeasibilityReport.md`

**Interfaces:**
- Consumes: Outputs from Task 1-4
- Produces: Complete Stage 0 deliverable covering compatibility matrix, acceleration, licenses, and Stage 1 risks

- [x] **Step 1: Aggregate audit data, license matrix, and benchmark results**
- [x] **Step 2: Formulate explicit compatibility matrix for `all-in-one-infer` vs Demucs/Librosa/PyTorch**
- [x] **Step 3: Write `FeasibilityReport.md` following Stage 0 requirements**
- [x] **Step 4: Verify acceptance criteria from roadmap are satisfied**
