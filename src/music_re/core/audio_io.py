import contextlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Generator, Tuple, Union

import numpy as np
import soundfile as sf


def normalize_audio(data: np.ndarray, target_peak_db: float = -1.0) -> np.ndarray:
    """Normalize audio array to a target peak dBFS without changing time/phase."""
    if not np.all(np.isfinite(data)):
        raise ValueError("Audio data contains NaN or Inf values")
    if data.size == 0:
        raise ValueError("Audio data is empty/zero-length")
    target_amplitude = 10.0 ** (target_peak_db / 20.0)
    current_peak = float(np.max(np.abs(data)))
    if current_peak == 0.0:
        return data.astype(np.float32)
    scale = target_amplitude / current_peak
    return (data * scale).astype(np.float32)


@contextlib.contextmanager
def decode_to_temp_wav(
    input_path: Union[str, Path],
    target_sr: int | None = None,
) -> Generator[Path, None, None]:
    """Decode any supported audio format into a temporary uncompressed 32-bit float WAV."""
    src = Path(input_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found in PATH")

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_wav = Path(tmp_dir) / "decoded.wav"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
        ]
        if target_sr is not None:
            cmd.extend(["-ar", str(target_sr)])
        cmd.extend(["-c:a", "pcm_f32le", str(temp_wav)])

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg decoding failed for {src}: {res.stderr.strip()}")

        yield temp_wav


def decode_audio(
    input_path: Union[str, Path],
    target_sr: int | None = None,
    normalize_db: float | None = -1.0,
) -> Tuple[np.ndarray, int]:
    """Decode audio via FFmpeg to a numpy float32 array, with optional peak normalization."""
    with decode_to_temp_wav(input_path, target_sr=target_sr) as temp_wav:
        data, sr = sf.read(str(temp_wav), dtype="float32")

    if not np.all(np.isfinite(data)):
        raise ValueError(f"Audio file {input_path} contains NaN or Inf values")
    if data.size == 0:
        raise ValueError(f"Audio file {input_path} is empty or zero-length")

    if normalize_db is not None:
        data = normalize_audio(data, target_peak_db=normalize_db)

    return data, sr
