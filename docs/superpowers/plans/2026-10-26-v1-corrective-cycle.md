# MusicRE v1 Corrective Cycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Track each step with checkboxes. Do not treat mocked integration tests as real-model acceptance.

**Goal:** Address confirmed BUG-01 through BUG-14, then establish CLI/MCP parity and export completeness on a real MP3 with actual models.

**Execution outcome:** Tasks 1–8 completed sequentially. Final locked suite: 147 passed, 1 skipped; all three real-model proof runs passed. See `docs/verification/2026-10-26-corrective-cycle.md` for the executed evidence, review fixes, rulings, and limits. Step checkboxes below preserve the original planning text; the execution ledger records actual per-task completion. The user subsequently authorized committing and merging locally into `main`; remote publication remains outside scope.

**Architecture:** Keep the existing adapters and workspace. Introduce a shared pipeline service and explicit profile settings; fingerprint stage inputs and replace active artifacts when configuration changes. Use upstream evaluation libraries, nullable unknown confidence, and explicit transcription results rather than fabricated success.

**Tech Stack:** Python 3.11, uv, Pydantic, Typer, FastMCP, Demucs, All-In-One-Infer, Basic Pitch ONNX, Librosa, PrettyMIDI, mir_eval, museval, pytest, FFmpeg.

**Spec:** The user-supplied audit of revision `6fc8820`, covering P0/P1 and the eight-item corrective checklist. Research-stage model additions remain out of scope.

## Global Constraints

- Preserve user files, existing local reports, and cached model weights.
- Do not publish, push, or claim release readiness without actual acceptance evidence.
- Unknown confidence is `null`, not a constant. Baseline segmentation must not claim semantic labels or detected downbeats.
- Objective quality means at least one valid reference comparison actually ran; each metric group reports its own mode and coverage.
- Use library metrics with documented settings. Do not describe amplitude percentiles as SNR.
- Retain Python `>=3.11,<3.12`; update and track the dependency lock.
- Model tests may download weights. Record failures, unavailable resources, and skipped verification distinctly from passes.
- Prefer existing stack and interfaces; no new research models or custom objective metric algorithms.
- No automatic commits are required for this cycle. Keep changes available for review.

## Review Focus

1. Switching six/four stems or profiles must never retain old MIDI, evaluation, or completed-stage state.
2. Missing or corrupt cached artifacts must trigger regeneration, not a successful resume or incomplete export.
3. Empty reference directories, silent references, wrong-pitch notes, and corrupt MIDI must not produce false objective success.
4. Silent input must retain every decoded sample; boundary notes and variable-tempo grids must remain valid.
5. Cold CLI and MCP runs must exercise identical settings; warm runs alone do not prove model integration.

## Initial Evidence

- Local HEAD: `6fc8820`; working tree initially clean.
- Baseline command: `.venv/bin/python -m pytest -q`.
- Result: `96 passed, 5 warnings in 12.20s`.
- `uv.lock` exists locally but is untracked; `.gitignore` excludes `*.lock`, `*.json`, and `docs/superpowers/`.
- Environment contains Demucs 4.1.0, Basic Pitch 0.4.0, Torch 2.2.2, and mir_eval. Neither `allin1` nor `allin1_infer` is installed.
- Local feasibility/hardware/benchmark files exist but are untracked. They are historical evidence, not evidence for this cycle.
- No input files found in `data/`. A user-approved real MP3 or a redistributable public recording is required for acceptance.
- Upstream PyPI currently advertises `all-in-one-infer==3.1.0`; upstream `analyze()` accepts `stems_input` and `stem_provider`, not `demucs_dir`.
- References: https://pypi.org/project/all-in-one-infer/ and https://github.com/openmirlab/all-in-one-infer/blob/main/src/allin1_infer/analyze.py . Verify the installed pinned wheel as well; do not rely only on repository main.

## Task 1: Honest structure analysis and working upstream integration (BUG-01, BUG-02)

**Files:** `src/music_re/core/backends/{__init__,allin1_backend,librosa_structure_backend}.py`, `src/music_re/schemas.py`, `pyproject.toml`, `tests/test_structure_registry.py`, `tests/test_librosa_structure.py`; add `tests/test_allin1_backend.py`.

**Interfaces:** Preserve `analyze(audio_path, stems=None, device="auto") -> MusicalTimeline`. Allow `AllInOneStructureAnalyzer(model_name="harmonix-all")`. Change `TimelineSection.confidence` to `Optional[float] = None` with bounds `[0,1]`. Add timeline backend/model provenance and warnings with backward-compatible defaults.

- [ ] Add failing contract tests for `allin1_infer`, exact upstream argument names, stem reuse, empty result rejection, real audio duration, and absent segment confidence.
- [ ] Add failing tests asserting all Librosa labels equal `unknown`, confidence is `None`, and downbeats are empty rather than inferred by beat index. A segmentation failure returns one unknown full-length section with a warning.
- [ ] Run `.venv/bin/python -m pytest tests/test_structure_registry.py tests/test_librosa_structure.py tests/test_allin1_backend.py -q`; confirm failures describe the audited behavior.
- [ ] Pin a compatible All-In-One-Infer version after inspecting its installed API. Use a precomputed provider for the four required stems; retain supplied files and isolate upstream intermediates. With six stems, create a temporary combined `other` containing other/guitar/piano so the four-source representation does not lose energy. Test this composition.
- [ ] Import `allin1_infer` in adapter and registry. Preserve genuinely absent dependency fallback only for `auto`; explicit neural requests and inference failures must be errors, not silent Librosa fallback.
- [ ] Implement unknown semantics and provenance. Measure duration from decoded input rather than the last detected beat.
- [ ] Run targeted tests and `.venv/bin/python -m pytest -q`; adjust existing tests that asserted fabricated labels/confidence.

## Task 2: Duration preservation and configuration-aware active workspace (BUG-03, BUG-04, BUG-10)

**Files:** `src/music_re/core/backends/demucs_backend.py`, `src/music_re/core/separator.py`, `src/music_re/workspace.py`, `src/music_re/schemas.py`, `tests/test_demucs_backend.py`, `tests/test_workspace_cache.py`, `tests/test_pipeline_resumption.py`.

**Interfaces:** Add manifest `schema_version`, `run_config`, `runtime`, `stage_fingerprints`, and `transcription_results` dictionaries. Add `WorkspaceManager.invalidate_stages(file_path, stages) -> WorkspaceManifest`. Reuse existing stage methods, replacing active `stems`/`midis` mappings rather than merging unrelated runs.

- [ ] Add a failing silent separation test using multi-second stereo input; assert exact frame count, channels, sample rate, and zero output without loading weights.
- [ ] Add failing six-to-four-to-six tests asserting exact active stem keys and cleared dependent artifacts/stages. Add missing-artifact and changed-settings resume tests.
- [ ] Run targeted tests to establish the failures.
- [ ] Use `torch.zeros_like(wav_raw)`. Keep atomic writes and completion-marker cleanup.
- [ ] Implement stage invalidation for changed inputs/settings. Invalidate timeline, analysis dependent on stems, MIDI, and evaluation when separation changes; invalidate MIDI/evaluation when timeline or transcription settings change.
- [ ] Save selected model IDs, library versions, requested/resolved device, execution parameters, source content hash, and available weight identifiers/hashes. Old manifests remain readable but cannot resume stages without valid fingerprints.
- [ ] Register separation artifacts on cache hits as well as fresh separation; never let standalone CLI/MCP split leave an obsolete active manifest.
- [ ] Run targeted tests and the full suite.

## Task 3: Reliable MIDI grid, endings, and result states (BUG-07, BUG-08, BUG-12)

**Files:** `src/music_re/core/{midi_quantize,midi_tonal,midi_drums}.py`, `src/music_re/core/backends/{transcriber_base,basic_pitch_backend,drum_backend}.py`, `src/music_re/{schemas,cli,mcp_server}.py`; extend MIDI/backend/CLI tests.

**Interfaces:** Define `TranscriptionResult` with `status` (`notes` or `no_notes`), `midi_path`, optional `raw_midi_path`, `note_count`, and backend metadata. Adapter `transcribe(..., grid_subdivisions=4, quantize_ends=True) -> TranscriptionResult`; legacy convenience wrappers may still return the MIDI path. Runtime failures raise and are recorded as errors, never as `no_notes`.

- [ ] Add failing CLI/MCP propagation tests for grid 8 and onset-only mode.
- [ ] Add failing quantization tests for start/end snapping, strictly positive length, duplicate beats, invalid subdivisions, pickups before the first beat, variable tempo, and final-track clipping.
- [ ] Add failing Basic Pitch tests separating valid zero-note MIDI, missing output, corrupt output, and inference exceptions. Ensure failures cannot replace a previously valid MIDI with an empty file.
- [ ] Run targeted tests and confirm failures.
- [ ] Validate grid arguments, snap endings when requested, and ensure collapsed notes use the next valid grid point or documented boundary clipping. Preserve raw MIDI before quantization.
- [ ] Propagate settings through both adapters, wrappers, CLI, and MCP. Return explicit successful result states; write final outputs atomically and raise on missing/corrupt generated MIDI.
- [ ] Run MIDI, CLI/MCP, and full regression suites.

## Task 4: Library-backed, honest quality reports (BUG-06 and diagnostic naming)

**Files:** `src/music_re/core/quality.py`, `src/music_re/schemas.py`, `pyproject.toml`, `tests/test_quality_evaluation.py`, `tests/test_review_fixes_stage5.py`; add focused regression tests.

**Interfaces:** Keep `evaluate_quality(workspace, file_path, ground_truth_stems=None, ground_truth_midi=None) -> QualityReport`. Each group exposes `mode`, comparison coverage/counts, settings, and warnings. Top-level `is_objective` is true only if a library reference comparison completed successfully.

- [ ] Add failing tests for empty/unmatched reference directories, missing estimates, wrong pitch with identical onsets, duplicate combined MIDI counting, empty notes, silent references, and corrupt MIDI.
- [ ] Run tests to demonstrate current false positives.
- [ ] Use `museval` for documented separation SDR/SIR/SAR with fixed source mapping and no permutation. Reject or report incompatible durations/channels explicitly rather than silently truncating comparisons. Report silent/nonfinite results as unavailable, not fabricated caps or numeric success.
- [ ] Use `mir_eval.transcription` for pitch-plus-onset and pitch-plus-onset-plus-offset scores; label both accurately and document tolerances. Keep drums separate from pitched-note aggregate metrics, using onset/pitch-class matching appropriate to GM events.
- [ ] Exclude `all_stems` by artifact role, not filename alone, in every per-stem aggregate. Record diagnostic parse errors rather than ignoring them.
- [ ] Rename the percentile-based quantity to `rms_to_p10_amplitude_db`; handle empty/silent audio and strict JSON serialization without NaN/Infinity. Report reconstruction residual as diagnostic only.
- [ ] Update CLI metric labels and run targeted/full tests. Fix fixtures that create files without registering manifest artifacts; these must no longer prove objective success.

## Task 5: One resumable pipeline and explicit profiles (BUG-05, BUG-09, BUG-10)

**Files:** add `src/music_re/core/profiles.py`, `src/music_re/core/pipeline.py`; modify `src/music_re/{cli,mcp_server}.py`, `src/music_re/core/{separator}.py`, `src/music_re/core/backends/demucs_backend.py`; extend pipeline, resumption, and profile tests.

**Interfaces:** `resolve_profile(name, six_stems=False) -> PipelineSettings`; `run_pipeline(file_path, workspace=None, six_stems=False, device="auto", profile="balanced", progress_callback=None) -> dict`. Return existing MCP keys (`workspace_dir`, `analysis`, `timeline`, `stems`, `midis`, `quality_report`) plus provenance/results.

**Proposed settings (implementation choices, not measured quality guarantees):**

| Setting | fast | balanced | quality |
| --- | --- | --- | --- |
| Stem count | 4 | requested 4/6 | requested 4/6 |
| Demucs shifts | 0 | 1 | 2 |
| Demucs overlap | 0.1 | 0.25 | 0.5 |
| Structure | Librosa baseline | All-In-One `harmonix-fold0` | All-In-One `harmonix-all` |
| Raw MIDI | no | yes | yes |
| Grid subdivisions | 4 | 4 | 4 |
| Quantize endings | yes | yes | yes |
| Diagnostic evaluation | no | yes | yes |

- [ ] Add failing adapter propagation/profile validation tests. Unknown profiles must fail before writing workspace outputs.
- [ ] Add failing equivalence tests proving CLI and MCP delegate to the same service with the same settings.
- [ ] Add failing resumption tests for unchanged settings, changed profile/model/grid, corrupt/missing artifacts, partial transcription recovery, and evaluation failure stage attribution.
- [ ] Run tests and confirm failures.
- [ ] Implement profiles, pass separation settings to `apply_model`, and include every effective setting in cache fingerprints. Use stable per-run seeds and record them; do not claim cross-device bitwise determinism.
- [ ] Move orchestration to `run_pipeline`. Set current stage before work, persist each artifact/result, reload manifests after invalidation, use timeline BPM on fresh and resumed runs, and mark completion only after required evaluation.
- [ ] Make CLI/MCP thin entry points. Retain existing CLI exit behavior and MCP response shape; expose cache-root selection for independent acceptance runs.
- [ ] Run targeted/full tests; inspect manifest contents for both entry points.

## Task 6: Complete and portable export (BUG-11)

**Files:** `src/music_re/core/export.py`, `tests/test_export_engine.py`, `tests/test_review_fixes_stage5.py`, `tests/test_stage5_cli.py`.

**Interfaces:** Preserve `export_bundle(...) -> Path`; add a preflight validator that raises `ValueError` listing missing/invalid required artifacts before creating the destination. Export profile must agree with the recorded execution profile.

- [ ] Add failing tests for empty workspace, missing stem/MIDI/timeline/report, incomplete stage, corrupt files, profile mismatch, stale raw MIDI, and unsafe directory/ZIP destinations.
- [ ] Add tests for ZIP-normalized paths, source-cache ancestor destinations, and existing destinations. Refuse overwriting an existing bundle without explicit authorization; do not merge stale contents.
- [ ] Run targeted tests and verify failures.
- [ ] Require the active 4/6 stems, per-stem MIDI and explicit result states, combined MIDI, timeline, manifest, and analysis; require raw MIDI and quality report according to the recorded profile.
- [ ] Copy only registered active artifacts. Rewrite exported manifest artifact paths to bundle-relative paths and include analysis metadata. Preserve source provenance separately.
- [ ] Stage exports before publishing; clean failed temporary output without touching source/cache files. Validate the final ZIP path after suffix normalization as well as the original target.
- [ ] Run folder/ZIP tests and full suite.

## Task 7: Reproducible dependencies, documentation, and CI (BUG-13, BUG-14)

**Files:** `pyproject.toml`, `uv.lock`, `.gitignore`, `README.md`, `scripts/verify_hardware.py` or existing equivalent; add `.github/workflows/tests.yml` and tracked verification-report paths.

**Interfaces:** `uv sync --locked --extra dev` creates the documented environment; neural analysis and objective metrics dependencies are included in the locked dependency graph.

- [ ] Add checks for Apache-2.0 documentation/metadata, tracked lock eligibility, dependency imports, and valid documented profiles. Check actual existing script names before modifying them.
- [ ] Remove blanket lock exclusion or exempt `uv.lock`. Narrow JSON/report ignores instead of accidentally committing caches or user audio. Make this corrective plan eligible for tracking without exposing unrelated ignored files.
- [ ] Resolve/pin compatible dependencies and regenerate `uv.lock`. Verify the selected wheel API and transitive Demucs namespace compatibility in an isolated environment before replacing the working environment.
- [ ] Correct README license to Apache 2.0, install/test commands, profile table, unknown Librosa semantics, actual metric definitions, neural dependency behavior, and transcription result states. Verify model-weight license claims against primary sources rather than repeating unsupported dataset-license assumptions.
- [ ] Add CPU CI with Python 3.11, FFmpeg, locked dependencies, and the unit/contract suite. Keep real-model acceptance separately opt-in with declared asset/weight requirements.
- [ ] Run `uv lock --check`, locked-environment tests, and CLI/MCP import/help checks. Record exact commands/results.

## Task 8: Real MP3 acceptance, parity, and final review

**Files:** add `scripts/verify_real_e2e.py`, opt-in `tests/test_real_e2e.py`, and a tracked `docs/verification/2026-10-26-corrective-cycle.md` report; update this checklist.

**Interfaces:** Acceptance runner accepts an explicit MP3 input and separate CLI/MCP cache/output roots, invokes real entry points, and writes a machine-readable summary plus a human-readable evidence report. No model adapters may be mocked.

- [ ] Obtain the user's MP3 path or approval to use a redistributable public music recording. Record provenance, content hash, duration, and license; do not upload user audio.
- [ ] Add harness tests for failure propagation and detecting incomplete bundles, then implement the runner.
- [ ] Run cold CLI and MCP balanced-profile jobs against the same MP3 in independent caches with actual Demucs, All-In-One-Infer, and Basic Pitch. Exercise MCP via protocol/tool invocation, not only direct calls to Python functions.
- [ ] Compare settings, model versions, active artifact sets, decoded durations, timeline validity, MIDI validity/result states, combined track roles, manifest completion, diagnostics, and portable folder/ZIP exports. Use documented time tolerances rather than promising bit-identical MIDI.
- [ ] Run warm resumes and validate no unnecessary reruns. Run configuration-switch tests and silent-audio regressions.
- [ ] Exercise quality ensemble and fast profile on the real recording or a documented excerpt. An excerpt smoke run does not establish whole-recording quality or full-length acceptance.
- [ ] Re-run hardware/model and license checks; capture current timing/resource evidence. Preserve older local reports.
- [ ] Run the complete locked test suite, inspect `git diff --check`, and request an independent final code review. Fix findings with focused regressions.
- [ ] Publish local evidence identifying passed, failed, skipped, and blocked checks. Without ground truth, claim only operational E2E success and diagnostic evaluation, not measured transcription/separation accuracy or universal v1.0 readiness.

## Execution and Open Decisions

- Recommended execution: native/inline, one task at a time. These fixes share manifest and adapter interfaces; parallel edits would increase merge and contract risk.
- Input needed: real MP3 path, or permission to use a redistributable public recording.
- Installation and model downloads occur only after plan approval. If CPU, memory, network, or upstream compatibility blocks acceptance, document the blocker instead of substituting mocks.
- Final review may use a dedicated reviewer, as requested by the review workflow, after implementation and verification.
