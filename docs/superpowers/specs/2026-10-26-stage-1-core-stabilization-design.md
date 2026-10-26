# Stage 1: Core Stabilization Design Document

**Status:** Approved  
**Author:** AI + Human Partner  
**Target:** MusicRE Stage 1 (PR-02)

## 1. Intent & Purpose

Stabilize the existing MusicRE audio decoding, caching, analysis, and MIDI processing pipeline without introducing new machine learning models. Ensure deterministic, reproducible processing of local MP3/WAV/FLAC files with error observability and state resumption.

## 2. Requirements & Architecture Decisions

### 2.1 Unified Audio Decoding via FFmpeg
- Decode all incoming formats (MP3, WAV, FLAC, OGG, M4A) via `ffmpeg` subprocess to an uncompressed 32-bit float WAV (`-c:a pcm_f32le`) inside a controlled temporary directory.
- Read decoded audio using `soundfile.read(..., dtype='float32')` preserving native sample rate (or resampled to target 44.1 kHz for Demucs if needed) and exact duration in samples.
- Replace fragile `audioread` fallbacks that produce Python 3.13 deprecation warnings.

### 2.2 Audio Normalization & Time Alignment
- Peak normalization to -1.0 dBFS (scale factor `0.89 / max(abs(data))`) without altering time offsets, zero-padding, or audio length.
- Original duration in seconds and sample count must be preserved across all stems.

### 2.3 Content-Based Identifiers & Caching
- Replace filename-only identifiers with SHA-256 content hashes (first 16 hex chars) of the audio file.
- Cache segregation: separate cache directories/keys for 4-stem (`htdemucs`) and 6-stem (`htdemucs_6s`) separation models. A 4-stem cache entry must never be reused as a 6-stem entry.

### 2.4 Removal of 64-Beat Limitation
- Remove hardcoded cuts (such as `[:64]` in beat tracking or slicer) across `analyzer.py`, `slicer.py`, and `workspace.py`. Full track duration beats must be tracked and preserved.

### 2.5 Fix Key Detection on Drums
- Harmonic key estimation (music21/chroma) must only run on tonal stems (`bass`, `other`, `vocals` or the full mix) and explicitly skip `drums`.

### 2.6 MIDI Overlap, Deduplication & Raw Preservation
- Clean MIDI note collisions: merge identical pitches starting within a short tolerance window, resolve note ends when overlapping same pitch.
- Dual export: preserve raw unquantized MIDI (`stem_raw.mid`) alongside quantized/cleaned MIDI (`stem_quantized.mid`).

### 2.7 Pipeline Resumption & Error Handling
- Workspace status state machine: `pending` -> `decoding` -> `separating` -> `analyzing` -> `transcribing` -> `completed` (or `failed`).
- If an operation fails, write `stage_error` with actionable diagnostics to `manifest.json`.
- Allow rerunning a workspace without overwriting existing valid stage artifacts.

## 3. Non-Goals
- No new external ML model downloads in Stage 1.
- No DAW project file generation yet (Stage 5/6).
