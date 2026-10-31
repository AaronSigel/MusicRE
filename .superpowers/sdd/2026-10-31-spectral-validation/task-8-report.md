# Task 8 report

Status: incomplete. The benchmark implementation, benchmark unit tests, and remaining acceptance tests were not completed. No benchmark acceptance target has been evaluated.

R18 follow-up: added tested deterministic split membership and an initial paired F1 distribution / recall comparison helper. These helpers are not connected to the CLI or report. The helper evaluates only recall, and must not be treated as the complete acceptance assessment. R18 subprocess measurements, benchmark orchestration, remaining metrics and acceptance tests remain unimplemented. No helper subprocess test was run. `uv run --locked pytest tests/test_spectral_benchmark.py -q`: 2 passed after observing both tests fail for missing helpers. Full suite after these changes: 442 passed, 4 deselected, 13 warnings, matching the prior warning count.

Added `test_mcp_matches_pipeline_result_in_filter_mode`: compares the full MCP and pipeline results and checks that the second run makes no backend calls. It passes against the existing implementation; this is an acceptance test, not a red/green feature cycle.

Existing matrix coverage in `tests/test_spectral_pipeline_integration.py`:

- Off/baseline artifacts: `test_off_keeps_original_artifacts`, `test_legacy_manifest_reuses_upstream_fingerprints`.
- Audit and filter on four stems: `test_modes_and_cache_isolation`, `test_audit_report_contains_unscored_original_notes`.
- Threshold change: `test_config_change_only_reruns_spectral_and_evaluation`.
- Corrupt report: `test_corrupt_report_reruns_stage`, `test_failed_corrupt_report_revalidation_is_not_completed`.
- Failure handling: `test_validator_error_is_recorded`.
- DAWproject validated notes: `test_filter_dawproject_contains_validated_notes`.
- CPU execution uses mocked backends in the integration tests.

Verification: `uv run --locked pytest tests/test_spectral_e2e_acceptance.py -q`: 1 passed. `uv run --locked pytest -m "not models and not real_e2e and not dataset" -q`: 440 passed, 4 deselected, 13 warnings. No baseline warning comparison was available.

R18 defines isolated warm subprocess RSS and a cold/warm timing protocol; these are not implemented yet. The spec does not quantify "severe per-stem regression"; do not claim a pass without a defined threshold.

Real models and datasets were not run. Controller commands:

```bash
uv run --locked pytest -m models
uv run --locked pytest -m real_e2e
uv run --locked python scripts/run_quality_benchmarks.py --spectral-validation off audit filter --split test
```

The benchmark command is pending implementation and currently will not accept those flags.

## Task 8a

Status: benchmark A/B orchestration wired into `scripts/run_quality_benchmarks.py` (R9, R18). Real models/datasets not run; no acceptance target has been evaluated on real data.

- CLI: `--spectral-validation {off,audit,filter}+` (default `off`), `--split tune|test|all` (default `all`, uses `track_in_split`). Default invocation keeps the existing in-process `run_reference_benchmarks` path and report shape; no `spectral_validation` key is written.
- A/B (`run_spectral_ab`): per track, on one cache root `<cache>/<track_id>`: cold `off` (T_off), warm `off`, then each other requested mode. Every run is a fresh child process: `python scripts/run_quality_benchmarks.py --_worker <json>` runs `run_pipeline(spectral_validation=mode)` + `evaluate_quality` and prints one `MUSIC_RE_BENCH_RESULT {...}` line with `quality_report`, `wall_time_sec`, `peak_rss_mb` (child ru_maxrss), `error`, `per_stem_f1`. In A/B mode `reference_tracks` is built from the cold `off` runs.
- Per-stem F1: computed in the worker by scoring each active stem MIDI against the ground-truth instrument with the same group name (onset+pitch, `quality_midi._score`). Stems with no same-named reference group (e.g. piano/guitar in 6-stem mode) are not scored. This function is not covered by a unit test (needs a real manifest); a scoring failure is recorded as `per_stem_error`, leaving per-stem NOT_MEASURED.
- Report (`summarize_spectral_ab`, JSON key `spectral_validation` + Markdown section "Spectral validation A/B"): per mode pitched P/R/F1, per-stem F1, absent-instrument false notes, mean wall time, max peak RSS; per comparison mean deltas, per-track ΔF1 min/median/max, count of tracks with recall drop > 0.02, per-track rows, acceptance table.
- Acceptance definitions (paired tracks where both warm `off` and the mode succeeded; no data -> `NOT_MEASURED`):
  - f1 / precision: mean Δ >= 0.
  - recall: mean Δrecall >= -0.02 (0.02 PASS, 0.021 FAIL).
  - absent_false_notes: (off - mode)/off >= 0.20 over tracks with an `absent` group; off total 0 -> NOT_MEASURED.
  - per_stem ("severe" defined here): no stem's mean per-stem ΔF1 < -0.05.
  - extra_time: Σ(T_mode_warm - T_off_warm) / ΣT_off_cold <= 0.15.
  - extra_ram: max per-track RSS(mode) - RSS(warm off) <= 512 MiB.
  - sdr: max |ΔSDR| over average and per-stem SDR < 0.01 dB.
  - errors: any failed run of the mode -> FAIL.
- Fixed a latent crash: `_atomic_write_text` used `os`, `tempfile`, `contextlib` without importing them, so `main()` could not write reports.
- Worker test runs a real subprocess with `run_pipeline`/`evaluate_quality` replaced through `MUSIC_RE_BENCH_STUB=<module name>` (test-only hook in `worker_main`).

TDD: 7 new tests failed first (missing functions), then passed. `test_main_wires_default_and_ab_paths` was written after `main()` changes and passed on first run. `uv run --locked pytest tests/test_spectral_benchmark.py -q`: 11 passed. Full suite `uv run --locked pytest -m "not models and not real_e2e and not dataset" -q`: 451 passed, 4 deselected, 13 warnings (same warning count as before).

Real-run command (controller):

```bash
uv run --locked python scripts/run_quality_benchmarks.py --spectral-validation off audit filter --split test
```

Tune thresholds on `--split tune` first; report acceptance only from `--split test`.

## Task 8b

Status: remaining mocked-model CPU acceptance tests added to `tests/test_spectral_e2e_acceptance.py`. No production bugs surfaced; no production code changed. All new tests passed on first run (acceptance tests of existing behaviour, not red/green feature cycles). Real models/datasets not run.

New tests:

- `test_six_stem_audit_scores_nothing_and_keeps_original`: six stems, audit; every stem reported, drums `skipped`, all others `unscored`, no validated set, active = original.
- `test_six_stem_filter_rejects_guitar_but_never_piano_or_drums`: six stems, filter; guitar wrong-pitch note rejected, piano stays `unscored` with both notes kept, drums `skipped` and unchanged; validated set = 6 stems + `all_stems`; combined MIDI equals the union of validated stem notes.
- `test_identical_rerun_reuses_every_stage[audit|filter]`: second identical run makes zero separation/structure/transcription/analysis/evaluation calls, `SpectralMidiValidator.validate` is not re-invoked, and the result equals the first.
- `test_filter_dawproject_note_counts_match_validated_midis`: per-stem DAWproject note counts equal validated MIDI note counts; bass 2 -> 1.
- `test_spectral_validation_defaults_to_off_everywhere`: `run_pipeline` signature, CLI `deconstruct` (spied call), MCP `deconstruct_full_pipeline` (spied call), benchmark parser default `['off']`.

Stage 7 matrix (spec: "E2E matrix"):

| Row | Test |
| --- | --- |
| off == baseline | `test_off_keeps_original_artifacts`, `test_off_export_has_original_layout`, `test_legacy_manifest_reuses_upstream_fingerprints` (integration); `test_spectral_validation_defaults_to_off_everywhere` (acceptance) |
| audit, 4 stems | `test_modes_and_cache_isolation`, `test_audit_report_contains_unscored_original_notes` |
| filter, 4 stems | `test_modes_and_cache_isolation`, `test_quality_uses_filtered_midis`, `test_filter_export_includes_both_midi_sets_and_report` |
| 6 stems | `test_six_stem_audit_scores_nothing_and_keeps_original`, `test_six_stem_filter_rejects_guitar_but_never_piano_or_drums` |
| CPU | all mocked-model tests above run on CPU. Real-model CPU/CUDA full cycle: real run required (`pytest -m real_e2e`) |
| rerun uses cache | `test_identical_rerun_reuses_every_stage`, `test_mcp_matches_pipeline_result_in_filter_mode` (zero backend calls on second run) |
| threshold change reruns dependents only | `test_config_change_only_reruns_spectral_and_evaluation` |
| corrupt report rejected | `test_corrupt_report_reruns_stage`, `test_failed_corrupt_report_revalidation_is_not_completed` |
| validator failure in manifest | `test_validator_error_is_recorded` |
| DAWproject export consistent | `test_filter_dawproject_contains_validated_notes`, `test_filter_dawproject_note_counts_match_validated_midis` |
| MCP == pipeline | `test_mcp_matches_pipeline_result_in_filter_mode` |
| Real-model quality / timing / RAM acceptance | real run required (benchmark command above, `--split test`) |
