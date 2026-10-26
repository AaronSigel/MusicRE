# Stage 3: Musical Structure Analysis Design Document

**Status:** Approved  
**Author:** AI + Human Partner  
**Target:** MusicRE Stage 3 (PR-04)

## 1. Intent & Purpose

Extract complete musical structure from audio: tempo (BPM), beat grid, bar boundaries (downbeats), and structural sections ("intro", "verse", "chorus", "bridge", "outro") into a unified `timeline.json` without re-running stem separation.

## 2. Requirements & Architecture Decisions

### 2.1 Abstract Interface: `StructureAnalyzerBackend`
Create `src/music_re/core/backends/structure_base.py`:
- `backend_name: str`
- `analyze(audio_path: Path | str, stems: Optional[Dict[str, Path | str]] = None, device: str = "auto") -> MusicalTimeline`

### 2.2 Standard Timeline Schema: `MusicalTimeline`
In `src/music_re/schemas.py`:
- `TimelineSection`:
  - `label: str` (e.g. "intro", "verse", "chorus", "solo", "bridge", "outro")
  - `start_sec: float`
  - `end_sec: float`
  - `confidence: float`
- `MusicalTimeline`:
  - `duration_sec: float`
  - `bpm: float`
  - `beats: List[float]` (all timestamps in seconds)
  - `downbeats: List[float]` (bar onsets / 1st beat of each measure)
  - `sections: List[TimelineSection]`

### 2.3 Concrete Backends
1. **`AllInOneStructureAnalyzer`** (`src/music_re/core/backends/allin1_backend.py`):
   - Integrates `all-in-one-infer` Python API.
   - Accepts pre-separated Demucs stems from Stage 2 (`stems/htdemucs`) so Demucs separation is not duplicated.
   - Returns structured `MusicalTimeline`.
2. **`LibrosaStructureAnalyzer`** (`src/music_re/core/backends/librosa_structure_backend.py`):
   - Fast, dependency-light structure analyzer using Librosa's beat tracking and spectral novelty/recurrence segmentation (laplacian / checkerboard segmentation).
   - Serves as baseline and transparent fallback when heavy neural checkpoints are omitted.

### 2.4 Registry & Auto-Selection Factory
In `src/music_re/core/backends/__init__.py`:
- `get_structure_analyzer(name: str = "auto", device: str = "auto") -> StructureAnalyzerBackend`
- If name is "auto", tries All-In-One-Infer; if not available or fails to initialize, gracefully uses Librosa fallback.

### 2.5 Workspace & CLI Integration
- `timeline.json` saved in `<workspace>/timeline.json`.
- `WorkspaceManifest` updated with `"structure"` stage.
- CLI adds `music-re structure <audio>` command and integrates structure extraction into `music-re deconstruct`.

## 3. Non-Goals
- No retraining of neural models.
- No subjective human-in-the-loop manual section labeling.
