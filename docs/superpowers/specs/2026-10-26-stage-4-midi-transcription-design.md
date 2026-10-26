# Stage 4: MIDI Transcription & Time Synchronization Design Document

**Status:** Approved  
**Author:** AI + Human Partner  
**Target:** MusicRE Stage 4 (PR-05)

## 1. Intent & Purpose

Extract musical note events from separated audio stems into synchronized MIDI parts aligned with the composition's real temporal structure (`timeline.json`), preserving both raw and quantized versions, generating both individual stem MIDIs and an integrated multi-track project MIDI file (`all_stems.mid`).

## 2. Requirements & Architecture Decisions

### 2.1 Abstract Interface: `TranscriberBackend`
Create `src/music_re/core/backends/transcriber_base.py`:
- `backend_name: str` (e.g. "basic_pitch", "multiband_drums")
- `mode: str` ("tonal" | "drums")
- `transcribe(audio_path: Path | str, output_midi_path: Path | str, timeline: Optional[MusicalTimeline] = None, bpm: float = 120.0, quantize: bool = True, save_raw: bool = True) -> Path`

### 2.2 Dynamic Timeline Quantization
In `src/music_re/core/midi_tonal.py`:
- `quantize_to_timeline(pm: pretty_midi.PrettyMIDI, timeline: MusicalTimeline, grid_subdivisions: int = 4)`:
  - Build fine-grained grid timestamps by subdividing each inter-beat interval in `timeline.beats` into `grid_subdivisions` (16th notes).
  - For each note onset, find the nearest grid timestamp.
  - Snap start to nearest grid point; ensure duration is at least `min_duration`.
  - Fallback: if `timeline` is None or has <2 beats, use static tempo grid `(60.0 / bpm) * 0.25`.

### 2.3 Concrete Backends
1. **`BasicPitchTranscriber`** (`src/music_re/core/backends/basic_pitch_backend.py`):
   - Tonal polyphonic transcriber wrapping Basic Pitch ONNX runtime session.
   - Saves unquantized raw MIDI (`<stem>_raw.mid`) and quantized MIDI (`<stem>.mid`).
   - Resolves same-pitch overlapping notes.
2. **`DrumTranscriber`** (`src/music_re/core/backends/drum_backend.py`):
   - Multi-band energy detection (Kick MIDI 36, Snare MIDI 38, Hi-Hat MIDI 42).
   - Snaps drum onsets to timeline beat grid.

### 2.4 Multi-Track MIDI Project Aggregator
In `src/music_re/core/midi_project.py`:
- `build_combined_midi_project(stem_midis: Dict[str, Path | str], output_path: Path | str, bpm: float = 120.0, timeline: Optional[MusicalTimeline] = None) -> Path`:
  - Combines `vocals`, `bass`, `drums`, `other` (and optional `guitar`, `piano`) into a single Standard MIDI File (SMF Type 1).
  - Assigns distinct GM instruments and names (e.g. Bass -> Fingered Bass #33, Drums -> Channel 10 / Drum instrument).
  - Embeds tempo map from `timeline`.
  - Saves as `all_stems.mid` in `<track_workspace>/midis/`.

### 2.5 Registry & Workspace/CLI Integration
- In `src/music_re/core/backends/__init__.py`:
  - `get_transcriber_backend(mode="tonal", name="auto") -> TranscriberBackend`
- In `src/music_re/workspace.py`:
  - Saves transcriber metadata into `manifest.json`.
- In `src/music_re/cli.py`:
  - `music-re to-midi <stem> [--timeline <path>] [--raw] [--grid 16]`
  - `music-re deconstruct` generates individual stem MIDIs, raw backups, and `all_stems.mid`.

## 3. Non-Goals
- No custom deep learning training for transcription.
- No automatic pitch-bend microtonal curve synthesis in Stage 4.
