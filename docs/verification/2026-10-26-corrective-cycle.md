# MusicRE corrective-cycle verification

Date: 2026-10-26. Base revision: `6fc8820`. Working branch: `fix/v1-corrective-cycle`.

## Final outcome

**Corrective cycle operational acceptance passed on the stated inputs.** All three final runs used real models through both CLI and stdio MCP, independent cold workspaces, successful warm resumption, matching MIDI results within 3 ms, exact decoded stem frame counts, and byte-identical folder/ZIP contents per interface.

| Profile / input | Cold CLI | Cold MCP | Result |
| --- | ---: | ---: | --- |
| balanced / complete 205.139592 s recording | 171.62 s | 186.48 s | passed |
| quality / 0–20 s excerpt, Harmonix 8-fold ensemble | 85.66 s | 58.21 s | passed |
| fast / same 20 s excerpt, Librosa baseline | 24.63 s | 22.41 s | passed |

Machine-readable evidence: [balanced full recording](2026-10-26-e2e-balanced.json), [quality excerpt](2026-10-26-e2e-quality-20s.json), [fast excerpt](2026-10-26-e2e-fast-20s.json).

The full balanced run produced 4 stems, 5 MIDI artifacts plus 4 raw MIDI files, 321 beats, 81 neural downbeats, 12 sections, and 5,146 per-stem MIDI notes (the combined MIDI is not counted twice). Its peak child-process RSS was 4,835,336 KiB on this Linux host. These counts are diagnostics, not accuracy claims. The quality ensemble was tested on an excerpt, not the complete recording.

## Scope and limits

This cycle addresses audit BUG-01 through BUG-14. Unit/contract tests are not evidence of musical accuracy. The real-model acceptance runner uses independent CLI and stdio MCP workspaces, invokes actual models, exports directories and ZIPs, and checks warm resumption. No reference stems or MIDI are available for the public recording, so its reports are diagnostic, not objective accuracy measurements.

## Input recording

- **Carefree**, Kevin MacLeod (incompetech.com).
- Download: https://incompetech.com/music/royalty-free/mp3-royaltyfree/Carefree.mp3
- Attribution/license page: https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1400037
- Licensed under Creative Commons: By Attribution 4.0, https://creativecommons.org/licenses/by/4.0/ . The publisher's license page provides this attribution form.
- SHA-256: `8433b770a630d9b1594fd484442c677907ece899a4d149954cd2e74fd733e311`.
- Container duration reported by FFprobe: 205.139625 seconds. Decoded PCM duration is checked separately by the runner; MP3 container padding is not treated as audio duration.
- Downloaded file is local at `/tmp/opencode/MusicRE-Carefree.mp3`; neither recording nor generated stems are committed.

## Commands and current evidence

```bash
uv sync --locked --extra dev
uv lock --check
.venv/bin/python -m pytest -q --tb=short --disable-warnings
git diff --check
OMP_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python scripts/verify_hardware_and_models.py
.venv/bin/python scripts/audit_licenses.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=3 \
  .venv/bin/python scripts/verify_real_e2e.py /tmp/opencode/MusicRE-Carefree.mp3 \
  --out /tmp/opencode/musicre-proof-balanced
```

- Initial suite: **96 passed**, 5 warnings.
- Pre-acceptance locked suite: **141 passed, 1 skipped**, 5 warnings, 31.39 seconds. The skip is the opt-in real MP3 test without `MUSIC_RE_TEST_MP3`; it is not counted as an acceptance pass.
- Lock validation and `git diff --check`: exit 0.
- Hardware/model probe: CPU, 16 logical cores, Torch threads set to 2, no available CUDA/ROCm. Demucs `htdemucs` and Basic Pitch ONNX loaded successfully. See [hardware report](2026-10-26-hardware.json).
- Installed package/license inventory: [license report](2026-10-26-licenses.json). Checkpoint commercial-use rights remain explicitly unverified where primary sources do not establish them. Repository license is Apache-2.0, not MIT.
- GitHub Actions workflow is authored; its execution on GitHub has not been observed here.
- First real MP3 attempt: the full CLI pipeline and folder/ZIP exports completed; cold CLI wall time was **194.60 seconds**, with 5,018 per-stem notes in its diagnostic report. MCP failed during decoding with `Disk quota exceeded`. The run as a whole is **failed**, not passed.
- The isolated environment had occupied 7.2 GiB on `/tmp` (a 12 GiB tmpfs). It was re-created under `/home/funder/.cache/musicre-v1-corrective-venv` with the same lock, and the worktree `.venv` now links to it. Only the environment created for this cycle was removed from tmpfs; the original project environment and old evidence were preserved.
- A second attempt (`/tmp/opencode/musicre-acceptance-balanced-2`) was intentionally stopped before applying review fixes, to avoid testing different source versions across CLI and MCP. It is not counted as a pass.
- The post-review `/tmp/opencode/musicre-acceptance-final-balanced` run completed both cold model pipelines, but the harness failed while reading the successful MCP result: this MCP version wraps generic return values in `structuredContent.result`. Cold CLI time was 197.79 seconds. This harness run remains recorded as **failed**. A protocol inspection confirmed `isError=False` and the expected nested result.
- The harness parser was corrected with a RED→GREEN regression test. The following full-recording run reached the final export comparison before revealing the serialization issue described below; its failure evidence remains preserved.
- Quality ensemble and fast-profile real-audio checks: **passed on the 20-second excerpt** in the final sequence.
- Independent final code review found three Important bugs: unsafe lower-level cache reuse, missing downstream invalidation after same-path regeneration, and checkpoint provenance changes on warm resume. All three were reproduced with failing tests and fixed. The onset-only ending issue was regraded as Important because it contradicted documented behavior, reproduced, and fixed in the same pass.
- Post-review complete suite: **145 passed, 1 skipped**, 5 warnings, 23.69 seconds. Lock and whitespace checks exit 0. No second review was substituted for the regression tests.
- After the harness parser regression: **146 passed, 1 skipped**, 5 warnings, 23.74 seconds.
- The next full run completed CLI (176.38 seconds), MCP (169.24 seconds), warm-resume snapshots, and result parity checks, but failed the final ZIP/folder byte check on JSON key ordering. Semantic equality was confirmed. Cross-process deterministic export was then fixed with a failing-then-passing regression using different `PYTHONHASHSEED` values.
- Latest complete suite after that fix: **147 passed, 1 skipped**, 5 warnings, 26.60 seconds. `uv build` produced both wheel and source distribution successfully; package version remains `0.1.0` and no release is implied.
- Final post-acceptance locked suite: **147 passed, 1 skipped**, 5 warnings, 25.00 seconds. `uv lock --check` and `git diff --check` both exited 0. The skipped opt-in test was exercised separately through the recorded real-model acceptance runs.
- Final proof sequence: fast excerpt, quality excerpt, then balanced full recording. Roots `/tmp/opencode/musicre-proof-fast-20s`, `/tmp/opencode/musicre-proof-quality-20s`, `/tmp/opencode/musicre-proof-balanced`. **All three passed.** Earlier failed attempts are not relabeled as passes.

## Fix coverage

| Audit items | Change | Evidence |
| --- | --- | --- |
| BUG-01/02 | Correct Infer API, stem reuse, unknown baseline semantics, nullable confidence | Adapter contracts and structure regressions; actual neural inference in full CLI attempt |
| BUG-03/04 | Exact silent sample length; replace active mappings and invalidate dependent state | Multi-second silence and 6/4 switching tests; standalone regeneration regression |
| BUG-05/09/10 | Shared pipeline, explicit profiles, package/device/checkpoint provenance and bound fingerprints | Shared CLI/MCP, resumption, profile and checkpoint replacement regressions |
| BUG-06 | Library reference metrics, comparison coverage, pitch-aware MIDI scores, no aggregate double count | Empty reference/wrong pitch/corrupt MIDI/silence regressions |
| BUG-07/08/12 | Propagated grid, optional ending quantization, explicit no-notes state, real errors | Quantization, CLI and output-preservation regressions |
| BUG-11 | Complete preflight, portable paths, active raw files only, exclusive ZIP publication | Missing/corrupt artifact, profile and path-safety tests |
| BUG-13/14 | Eligible lock, locked install, CI workflow, Apache-2.0 docs | Lock/dependency/import checks; CI execution still unverified |

## Decisions and residual limits

- The native worktree tool failed, so Git created the isolated branch/worktree directly. This preserves `main`; host-native worktree management may not own its cleanup.
- Runtime version/fingerprint integration was completed with the shared pipeline rather than twice in separate tasks. Cache invalidation is now covered by both low-level and orchestrator regressions.
- Undefined or infinite reference scores remain `null` with warnings rather than invented numeric caps. A user must inspect metric availability for perfect/infinite cases.
- Torch and Torchaudio are both pinned to 2.2.2 after the original resolution paired incompatible versions. Future maintenance/security upgrades require retesting the ML stack.
- Upstream Demucs repository licensing is MIT; checkpoint commercial-use rights remain unverified rather than inferred from training-dataset licenses. Commercial deployment needs a separate review.
- The cycle's environment moved off tmpfs; `.venv` links to `/home/funder/.cache/musicre-v1-corrective-venv`. Removing that environment will require recreating it.
- Mean-interval quantization-grid extrapolation after the last detected beat remains an estimate. Tempo changes after the final detection can cause tail drift.
- Actual-model acceptance is supported by the final runner evidence, not by mocked tests or the independent review alone. One recording and two excerpts do not establish general model accuracy.
- GitHub CI execution, remote push, release, and universal v1.0 readiness are not claimed. Following acceptance, the user authorized committing and merging locally into `main`. Remote publication remains outside scope.
- No review findings remain deliberately deferred as minor. The ending-preservation finding was regraded and fixed because it contradicted the documented flag contract.

## Reproduction

Use a fresh `--out` directory; the runner refuses an existing acceptance root. It writes subprocess logs and `result.json` even when inference fails. `status: failed` is not a successful or skipped acceptance run. The standard environment may download model weights on first use.

For an opt-in pytest invocation:

```bash
MUSIC_RE_TEST_MP3=/path/to/authorized-recording.mp3 \
  uv run --locked pytest -m real_e2e -q
```

The acceptance runner checks exact stem frame counts against the FFmpeg-decoded 44.1 kHz input, MIDI validity and counts, artifact sets, timeline duration, CLI/MCP parity, byte-identical ZIP/folder contents per interface, and that warm runs do not rewrite artifacts. This is an operational check on one recording, not a benchmark dataset or evidence of universal v1.0 readiness.
