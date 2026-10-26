import contextlib
import io
import os
from pathlib import Path
import shutil
import sys
from typing import Callable, Dict, Optional, Union

from music_re.schemas import StemSeparationResult
from music_re.workspace import WorkspaceManager

STEMS_4 = ["vocals", "drums", "bass", "other"]
STEMS_6 = ["vocals", "drums", "bass", "other", "guitar", "piano"]


def separate_stems(
    file_path: Union[str, Path],
    workspace: Optional[WorkspaceManager] = None,
    six_stems: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> StemSeparationResult:
    """Separate audio into stems using Demucs with cache inspection and stdio protection."""
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")

    if workspace is None:
        workspace = WorkspaceManager()

    track_dir = workspace.get_track_dir(p)
    stems_dir = workspace.get_stems_dir(p)
    model_name = "htdemucs_6s" if six_stems else "htdemucs"
    expected_stems = STEMS_6 if six_stems else STEMS_4

    # 1. Check cache: are all expected stems present and non-empty?
    cached_stems: Dict[str, str] = {}
    all_cached = True
    for stem in expected_stems:
        stem_path = stems_dir / f"{stem}.wav"
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
        )

    # 2. Run Demucs with protected stdout
    if progress_callback:
        progress_callback(f"Running Demucs ({model_name})...")

    # Demucs writes to output_dir/<model_name>/<track_stem>/
    tmp_demucs_out = track_dir / "demucs_tmp"
    tmp_demucs_out.mkdir(parents=True, exist_ok=True)

    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        from demucs.audio import AudioFile, save_audio
        import torch

        # Redirect stdout to stderr during Demucs model loading and separation
        with contextlib.redirect_stdout(sys.stderr):
            model = get_model(model_name)
            model.eval()

            wav = AudioFile(str(p)).read(
                streams=0,
                samplerate=model.samplerate,
                channels=model.audio_channels,
            )
            # Add batch dimension
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()
            sources = apply_model(model, wav[None], device="cuda" if torch.cuda.is_available() else "cpu")[0]
            sources = sources * ref.std() + ref.mean()

            stems_dict: Dict[str, str] = {}
            for name, source in zip(model.sources, sources):
                target_path = stems_dir / f"{name}.wav"
                save_audio(source.cpu(), str(target_path), samplerate=model.samplerate)
                stems_dict[name] = str(target_path)

    finally:
        if tmp_demucs_out.exists():
            shutil.rmtree(tmp_demucs_out, ignore_errors=True)

    return StemSeparationResult(
        workspace_dir=str(track_dir),
        model_used=model_name,
        stems=stems_dict,
    )
