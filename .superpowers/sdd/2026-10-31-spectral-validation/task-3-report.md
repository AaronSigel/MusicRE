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
- Pitch support and persistence use relative CQT energy, with STFT fundamental fallback outside CQT range. Harmonic bands are calculated but do not currently contribute to support; this limits harmonic-rich timbres and warrants follow-up calibration.
- Tests do not yet directly cover click rejection, decaying reverb tails, or sub-hop notes.
- No new dependencies. No validator class added, per task boundary.
