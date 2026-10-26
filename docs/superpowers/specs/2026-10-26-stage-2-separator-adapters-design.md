# Stage 2: Modular Separator Adapters Design Document

**Status:** Approved  
**Author:** AI + Human Partner  
**Target:** MusicRE Stage 2 (PR-03)

## 1. Intent & Purpose

Transform the audio separation logic from a tightly coupled Demucs script into a modular, extensible adapter architecture (`SeparatorBackend`). Provide deterministic device management (CPU, CUDA, ROCm), memory reuse/unloading, explicit 4-stem and 6-stem handling, and disk-based stem artifacts in the workspace.

## 2. Requirements & Architecture Decisions

### 2.1 Abstract Interface: `SeparatorBackend`
Create `src/music_re/core/backends/separator_base.py`:
- `backend_name: str` (e.g. "demucs")
- `model_name: str` (e.g. "htdemucs", "htdemucs_6s")
- `supported_stems: List[str]`
- `device: str` ("auto" -> detects CUDA/ROCm or falls back to CPU)
- `load() -> None`: Lazily loads weights into memory.
- `unload() -> None`: Drops model references and triggers GC / CUDA cache clearing.
- `separate(input_audio: Path | str, output_dir: Path | str, progress_callback: Callable[[str], None] | None = None) -> Dict[str, Path]`: Separates audio and writes validated WAV files to disk.

### 2.2 Concrete Demucs Adapter: `DemucsSeparator`
Create `src/music_re/core/backends/demucs_backend.py`:
- Implements `SeparatorBackend` wrapping `htdemucs` and `htdemucs_6s`.
- 4-stem mode is default (`vocals`, `drums`, `bass`, `other`).
- 6-stem mode explicitly documents bleed and quality constraints (`guitar` and `piano` have higher leakage).
- Support model instance reuse when called sequentially without re-downloading or re-instantiating.

### 2.3 Backend Registry & Factory
Create `src/music_re/core/backends/__init__.py`:
- `get_separator_backend(name: str = "demucs", model_name: str | None = None, six_stems: bool = False, device: str = "auto") -> SeparatorBackend`
- Validates backend existence and allows future alternatives (such as BS-RoFormer or MSST) to plug in cleanly.

### 2.4 Device & Memory Management
- "auto" mode inspects `torch.cuda.is_available()` and `torch.version.hip`.
- Explicit device overrides: `"cpu"`, `"cuda"`, `"cuda:0"`.
- Support explicit memory release after job completion.

### 2.5 Refactored `src/music_re/core/separator.py`
- Refactor `separate_stems` to delegate to `get_separator_backend(...)`.
- Preserve full backwards compatibility for existing CLI and tests.
- Stems continue to be saved under `<track_workspace>/stems/<model_name>/` as established in Stage 1.

## 3. Non-Goals
- No new ML model dependencies in Stage 2 (Demucs remains the primary separation engine).
- No direct in-memory tensor pipeline (WAV files on disk chosen for stability and low RAM).
