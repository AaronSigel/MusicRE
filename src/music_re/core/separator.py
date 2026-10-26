from pathlib import Path
from typing import Callable, Dict, Optional, Union

from music_re.core.backends import get_separator_backend
from music_re.schemas import StemSeparationResult
from music_re.workspace import WorkspaceManager

STEMS_4 = ["vocals", "drums", "bass", "other"]
STEMS_6 = ["vocals", "drums", "bass", "other", "guitar", "piano"]


def separate_stems(
    file_path: Union[str, Path],
    workspace: Optional[WorkspaceManager] = None,
    six_stems: bool = False,
    device: str = "auto",
    backend_name: str = "demucs",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> StemSeparationResult:
    """Separate audio into stems using pluggable SeparatorBackend adapter with caching."""
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")

    if workspace is None:
        workspace = WorkspaceManager()

    track_dir = workspace.get_track_dir(p)
    backend = get_separator_backend(
        name=backend_name,
        six_stems=six_stems,
        device=device,
    )
    model_name = backend.model_name
    expected_stems = backend.supported_stems
    quality_note = getattr(backend, "quality_note", None)
    stems_dir = workspace.get_stems_dir(p, model_name=model_name)

    # 1. Check cache: segregated model directory first, fallback to legacy stems_dir
    candidate_dirs = [stems_dir, track_dir / "stems"]
    for c_dir in candidate_dirs:
        completion_marker = c_dir / f".complete_{model_name}"
        if completion_marker.is_file():
            cached_stems: Dict[str, str] = {}
            all_cached = True
            for stem in expected_stems:
                stem_path = c_dir / f"{stem}.wav"
                if stem_path.is_file() and stem_path.stat().st_size > 0:
                    cached_stems[stem] = str(stem_path)
                else:
                    all_cached = False
                    break

            if all_cached:
                if progress_callback:
                    progress_callback("Stems retrieved from cache.")
                return StemSeparationResult(
                    workspace_dir=str(track_dir),
                    model_used=model_name,
                    stems=cached_stems,
                    quality_note=quality_note,
                )

    # 2. Run backend separation
    stems_res = backend.separate(
        input_audio=p,
        output_dir=stems_dir,
        progress_callback=progress_callback,
    )

    stems_dict = {name: str(path) for name, path in stems_res.items()}
    return StemSeparationResult(
        workspace_dir=str(track_dir),
        model_used=backend.model_name,
        stems=stems_dict,
        quality_note=quality_note,
    )
