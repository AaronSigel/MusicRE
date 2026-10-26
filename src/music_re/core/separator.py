from pathlib import Path
from typing import Callable, Dict, Optional, Union

from music_re.core.backends import get_separator_backend
from music_re.core.separation_cache import cache_context, cache_is_valid, save_cache_record
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
    shifts: int = 1,
    overlap: float = .25,
    force: bool = False,
    seed: int = 0,
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
        shifts=shifts,
        overlap=overlap,
    )
    model_name = backend.model_name
    expected_stems = backend.supported_stems
    quality_note = getattr(backend, "quality_note", None)
    cache_name = model_name if (shifts, overlap) == (1, .25) else f'{model_name}_s{shifts}_o{overlap:g}'
    stems_dir = workspace.get_stems_dir(p, model_name=cache_name)

    # 1. Check cache: segregated model directory first, fallback to legacy stems_dir
    candidate_dirs = [stems_dir] + ([track_dir / 'stems'] if cache_name == model_name else [])
    context = cache_context(p, backend, shifts, overlap, seed)
    for c_dir in ([] if force else candidate_dirs):
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

            if all_cached and cache_is_valid(completion_marker, cached_stems, context):
                if progress_callback:
                    progress_callback("Stems retrieved from cache.")
                workspace.mark_stage_complete(p, 'separation', artifacts=cached_stems)
                return StemSeparationResult(
                    workspace_dir=str(track_dir),
                    model_used=model_name,
                    stems=cached_stems,
                    quality_note=quality_note,
                )

    # 2. Run backend separation
    workspace.invalidate_stages(p, ['separation', 'analysis', 'structure', 'transcription', 'evaluation'])
    workspace.update_manifest(p, current_stage='separation')
    try:
        import random
        import torch
        random.seed(seed)
        torch.manual_seed(seed)
        stems_res = backend.separate(input_audio=p, output_dir=stems_dir, progress_callback=progress_callback)
        if set(stems_res) != set(expected_stems):
            raise ValueError('Separation output stem set does not match the selected backend')
        save_cache_record(stems_dir / f'.complete_{model_name}', stems_res,
                          cache_context(p, backend, shifts, overlap, seed))
    except Exception as e:
        workspace.record_stage_error(p, 'separation', str(e))
        raise
    finally:
        backend.unload()

    stems_dict = {name: str(path) for name, path in stems_res.items()}
    workspace.mark_stage_complete(p, 'separation', artifacts=stems_dict)
    return StemSeparationResult(
        workspace_dir=str(track_dir),
        model_used=backend.model_name,
        stems=stems_dict,
        quality_note=quality_note,
    )
