# Stage 5: Quality Evaluation, Export & MusicRE v1.0 Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement two-tier quality evaluation (`quality_report.json`), bundle export engine, execution profiles (`fast`, `balanced`, `quality`), MCP tools, and package MusicRE v1.0.

**Architecture:** Two-tier evaluation in `quality.py` (objective SDR/F1 with ground truth, diagnostic metrics without), bundle export in `export.py`, execution profiles in CLI, expanded MCP server tools, and updated v1.0 documentation.

**Tech Stack:** Python 3.11, soundfile, pretty_midi, scipy, numpy, typer, mcp.

**Spec:** `docs/superpowers/specs/2026-10-26-stage-5-quality-export-v1-design.md`

## Global Constraints

- Python strictly pinned to `>=3.11,<3.12`.
- Zero new external ML models; use existing stack.
- Never fabricate subjective percentage scores when ground truth is absent (`is_objective` must be strictly tracked).
- All existing 87 tests must continue to pass without regression.

## Review Focus

- Objective vs diagnostic metrics confusion in quality reports.
- Path traversal or overwriting source files during project export.
- Missing files in exported bundle (e.g. absent `all_stems.mid` or `timeline.json`).
- Fast vs quality profile parameter misconfiguration.
- MCP server tool schema divergence from CLI arguments.

---

### Task 1: Quality Report Schema & Two-Tier Evaluation Engine

**Files:**
- Modify: `src/music_re/schemas.py`
- Create: `src/music_re/core/quality.py`
- Modify: `src/music_re/workspace.py`
- Test: `tests/test_quality_evaluation.py`

**Interfaces:**
- Produces: `QualityReport` schema
- Produces: `evaluate_quality(workspace, file_path, ground_truth_stems=None, ground_truth_midi=None) -> QualityReport`
- Produces: `workspace.save_quality_report(...)` and `quality_report.json`

- [ ] **Step 1: Write failing test in `tests/test_quality_evaluation.py` for objective and diagnostic evaluation**
- [ ] **Step 2: Run `uv run pytest tests/test_quality_evaluation.py` and verify FAIL**
- [ ] **Step 3: Implement `QualityReport` in `schemas.py`, evaluation logic in `quality.py`, and save methods in `workspace.py`**
- [ ] **Step 4: Run `uv run pytest tests/test_quality_evaluation.py` and verify PASS**

---

### Task 2: Project Export Engine

**Files:**
- Create: `src/music_re/core/export.py`
- Test: `tests/test_export_engine.py`

**Interfaces:**
- Produces: `export_bundle(workspace, file_path, target_dir, profile="balanced", archive=False) -> Path`

- [ ] **Step 1: Write failing test in `tests/test_export_engine.py` validating exported directory structure and zip archive**
- [ ] **Step 2: Run `uv run pytest tests/test_export_engine.py` and verify FAIL**
- [ ] **Step 3: Implement `src/music_re/core/export.py` copying stems, midis, timeline.json, manifest.json, quality_report.json**
- [ ] **Step 4: Run `uv run pytest tests/test_export_engine.py` and verify PASS**

---

### Task 3: Execution Profiles & CLI Commands Extension

**Files:**
- Modify: `src/music_re/cli.py`
- Test: `tests/test_stage5_cli.py`

**Interfaces:**
- Produces: CLI commands `music-re evaluate` and `music-re export`
- Produces: CLI option `--profile [fast|balanced|quality]` in `deconstruct` and `export`

- [ ] **Step 1: Write failing test in `tests/test_stage5_cli.py` testing evaluate and export CLI commands**
- [ ] **Step 2: Run `uv run pytest tests/test_stage5_cli.py` and verify FAIL**
- [ ] **Step 3: Implement CLI commands `evaluate` and `export` and profile wiring in `src/music_re/cli.py`**
- [ ] **Step 4: Run `uv run pytest tests/test_stage5_cli.py` and verify PASS**

---

### Task 4: MCP Server Updates & Tool Registration

**Files:**
- Modify: `src/music_re/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

**Interfaces:**
- Produces: MCP tools `evaluate_quality`, `export_project`, `get_timeline`, `deconstruct_track`

- [ ] **Step 1: Update `tests/test_mcp_server.py` to assert new MCP tools are registered and functional**
- [ ] **Step 2: Run `uv run pytest tests/test_mcp_server.py` and verify FAIL**
- [ ] **Step 3: Implement tool handlers in `src/music_re/mcp_server.py`**
- [ ] **Step 4: Run `uv run pytest tests/test_mcp_server.py` and verify PASS**

---

### Task 5: End-to-End Release Validation & MusicRE v1.0 Documentation

**Files:**
- Create: `tests/test_v1_e2e.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Complete v1.0 pipeline
- Produces: 100% green full test suite and complete documentation

- [ ] **Step 1: Write comprehensive end-to-end release test in `tests/test_v1_e2e.py`**
- [ ] **Step 2: Run `uv run pytest tests/test_v1_e2e.py` and verify PASS**
- [ ] **Step 3: Update `README.md` with CLI and MCP guide, architecture, and licenses**
- [ ] **Step 4: Run full pytest suite across all tests (87+ tests) and verify 100% green**
