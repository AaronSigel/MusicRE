import contextlib
import io
import os
from pathlib import Path
import shutil
import sys
from typing import Callable, Dict, Optional, Union
import soundfile as sf

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
    """Separate audio into stems using Demucs with cache inspection, atomic writes, and stdio protection."""
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")

    if workspace is None:
        workspace = WorkspaceManager()

    track_dir = workspace.get_track_dir(p)
    stems_dir = workspace.get_stems_dir(p)
    model_name = "htdemucs_6s" if six_stems else "htdemucs"
    expected_stems = STEMS_6 if six_stems else STEMS_4
    completion_marker = stems_dir / f".complete_{model_name}"

    # 1. Check cache: are all expected stems present and completion marker exists?
    cached_stems: Dict[str, str] = {}
    if completion_marker.is_file():
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

    # 2. Check for silent audio early to avoid division by zero / NaN assertion errors
    info = sf.info(str(p))
    sr = info.samplerate
    channels = info.channels

    from demucs.audio import AudioFile, save_audio
    import torch

    wav_raw = AudioFile(str(p)).read(
        streams=0,
        samplerate=44100,
        channels=2,
    )

    if wav_raw.abs().max() < 1e-5:
        # Silent audio: write zero-filled stems atomically
        stems_dict: Dict[str, str] = {}
        for name in expected_stems:
            target_path = stems_dir / f"{name}.wav"
            tmp_path = stems_dir / f"{name}_tmp.wav"
            save_audio(torch.zeros_like(wav_raw), str(tmp_path), samplerate=44100)
            tmp_path.replace(target_path)
            stems_dict[name] = str(target_path)

        completion_marker.write_text("ok")
        return StemSeparationResult(
            workspace_dir=str(track_dir),
            model_used=model_name,
            stems=stems_dict,
        )

    # 3. Run Demucs with protected stdout
    if progress_callback:
        progress_callback(f"Running Demucs ({model_name})...")

    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    try:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                model = get_model(model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Demucs model '{model_name}'. Check network connection: {e}"
                ) from e

            model.eval()

            wav = AudioFile(str(p)).read(
                streams=0,
                samplerate=model.samplerate,
                channels=model.audio_channels,
            )
            ref = wav.mean(0)
            std = ref.std()
            if std < 1e-8:
                std = torch.tensor(1.0)
            wav = (wav - ref.mean()) / std
            sources = apply_model(model, wav[None], device="cuda" if torch.cuda.is_available() else "cpu")[0]
            sources = sources * std + ref.mean()

            stems_dict = {}
            for name, source in zip(model.sources, sources):
                target_path = stems_dir / f"{name}.wav"
                tmp_path = stems_dir / f"{name}_tmp.wav"
                save_audio(source.cpu(), str(tmp_path), samplerate=model.samplerate)
                tmp_path.replace(target_path)
                stems_dict[name] = str(target_path)

            completion_marker.write_text("ok")

    except Exception:
        if completion_marker.exists():
            completion_marker.unlink()
        raise

    return StemSeparationResult(
        workspace_dir=str(track_dir),
        model_used=model_name,
        stems=stems_dict,
    )
