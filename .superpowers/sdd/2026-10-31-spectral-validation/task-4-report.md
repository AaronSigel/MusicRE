# Task 4 report: spectral MIDI validation

## Built

- Added `RoleRule`, deterministic `SpectralValidatorConfig`, and `SpectralMidiValidator`.
- Audit computes evidence and marks each note `unscored`; it never writes MIDI.
- Filter evaluates once per note against stem features, removes only rejects, and preserves source MIDI.
- Drums skip; piano/vocals stay audit-only; unsupported stems warn and copy through.
- Added tests for audit bytes, wrong-pitch rejection, preservation, polyphony, duplicates, empty MIDI, role handling, and argument errors.

## Thresholds

- Bass/guitar: reject below harmonic support `0.10` and persistence `0.20`; uncertain below `0.10` or `0.20`.
- Other: stricter reject at support `0.06` and persistence `0.10`; uncertainty at support `0.10` or persistence `0.20`.
- Borderline test uses wrong C4 evidence on 440 Hz: bass rejects it; `other` marks it uncertain because its reject cutoffs are lower.
- Observed correct A4 support `0.7993`, quiet A4 `0.7991`, vibrato/high-note cases above conservative reject support; wrong C4 `0.0213` with persistence `0.0`.
- Bass MIDI 33 correct support `0.7820`; wrong neighboring tone support `0.4245`. The conservative reject conjunction avoids discarding it.

## Verification

- RED: targeted test collection failed as expected because `RoleRule` was absent.
- RED refinement: observed decision ordering follows MIDI-decoded note order; test now keys decisions by pitch.
- GREEN: initial validator tests — 5 passed.
- Expanded validator tests now have distinct named cases for all 10 requested areas, plus the `other` threshold comparison.

## Concerns

- Defaults are conservative and experimental; `note_score` remains null. Filtering stays opt-in.
- PrettyMIDI serialization decodes event times at MIDI tick precision; tests compare kept events to source-decoded values.

## Completion

- Validator tests: `uv run --locked pytest tests/test_midi_validator.py -q` — 15 passed.
- Full suite: `uv run --locked pytest -m "not models and not real_e2e and not dataset" -q` — 408 passed, 4 deselected, 13 warnings.
