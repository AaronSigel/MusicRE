# Task 3 report: per-note spectral evidence

## Built

- Added `NoteEvidence`, `StemValidationReport`, and `SpectralValidationResult` schemas.
- Added `compute_note_evidence(features, pitch, start_sec, end_sec, velocity=100)` using precomputed `StemSpectralFeatures`.
- Added synthetic tests for pitch discrimination, octave ambiguity, polyphony, bass-neighbour discrimination, vibrato, gain invariance, silence, and audio boundaries.

## Test evidence

- RED: initial test collection failed as expected because `music_re.core.midi_validation` did not exist.
- GREEN: `uv run --locked pytest tests/test_note_evidence.py -v` — 3 passed.
- Full suite: `uv run --locked pytest -m "not models and not real_e2e and not dataset" -q` — 366 passed, 4 deselected.
- Full suite printed 13 existing dependency/pipeline warnings; targeted tests were warning-free.

## Observed feature values

Values use `(harmonic_support, spectral_contrast, temporal_persistence, leakage)`:

- A4 on 440 Hz: `(1.0, 0.99999, 1.0, 0.0)`.
- C4 on 440 Hz: `(0.000017, 0.00704, 0.0, 0.9970)`.
- A5 on 440 Hz: `(0.000064, 0.00086, 0.0, 0.99987)`.
- MIDI 33 on 55 Hz: `(1.0, 0.999997, 1.0, 0.0)`.
- MIDI 33 on 58.27 Hz: `(0.48975, 0.999997, 1.0, 0.34255)`.

## Self-review and concerns

- `note_score` remains `None`; decisions remain `unscored`.
- Pitch support blends relative CQT energy with STFT harmonic-band prominence. Persistence uses a stem-wide expected-pitch threshold to make decaying tails less persistent.
- `note_score` and decisions remain uncalibrated as required.
- No new dependencies. No validator class added, per task boundary.

## Completion

- Added separate tests for correct/wrong pitch, octave leakage, two-note polyphony, 55 Hz versus 58.27 Hz bass, -40 dB gain, ±30-cent/5 Hz vibrato, reverb tail, onset click, past-end clipping, silence, out-of-range notes, zero-length/sub-hop windows, and bounded finite features.
- Added harmonic-tone test. Harmonic A4 support exceeds pure A4 support and remains above wrong-pitch support.
- RED: new reverb-tail test failed because local-only persistence marked every decaying frame present. Switched persistence threshold to stem-wide expected-pitch level; test passed.
- RED: harmonic support test failed with saturated/ineffective harmonic contribution. Added STFT prominence for the first four harmonics; test passed.
- Focused: `uv run --locked pytest tests/test_note_evidence.py -q` — 26 passed.
- Full suite: `uv run --locked pytest -m "not models and not real_e2e and not dataset" -q` — 389 passed, 4 deselected, 13 existing warnings.

Observed values `(harmonic_support, spectral_contrast, temporal_persistence, leakage)`:

- A4 on 440 Hz: `(0.7993, 1.0, 1.0, 0.0)`; wrong C4: `(0.0213, 0.0070, 0.0, 0.9970)`.
- A5 octave error: `(0.0379, 0.0009, 0.0, 0.9999)`.
- A4 on 440+880+1320 Hz: `(0.9299, 1.0, 1.0, 0.0)`.
- Quiet A4 at -40 dB: `(0.7991, 1.0, 1.0, 0.0)`.
- 55 Hz / MIDI 33: `(0.7820, 1.0, 1.0, 0.0)`; 58.27 Hz / MIDI 33: `(0.4245, 1.0, 1.0, 0.3426)`.
- Polyphonic 440+523.25 Hz: A4 support `0.7889`, C5 support `0.7336`; both persistence `1.0`.
- Reverb-like tail persistence `0.1143`; sustained A4 persistence `1.0`.
- Silence and fully out-of-range window: all five features `0.0`; out-of-range reason `out_of_range`.
