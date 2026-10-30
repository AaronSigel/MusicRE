import os
from pathlib import Path
from typing import Optional, Union
import soundfile as sf


def slice_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    start_sec: float,
    end_sec: Optional[float] = None,
) -> Path:
    """Extract slice from audio file using frame seeking with soundfile.

    Returns the output_path as a Path object.
    """
    in_p = Path(input_path).resolve()
    out_p = Path(output_path).resolve()

    if not in_p.is_file():
        raise FileNotFoundError(f"Input audio file not found: {in_p}")

    if in_p == out_p or (out_p.exists() and os.path.samefile(in_p, out_p)):
        raise ValueError("output_path cannot be identical to input_path")

    if start_sec < 0:
        raise ValueError(f"start_sec cannot be negative: {start_sec}")

    if end_sec is not None and end_sec <= start_sec:
        raise ValueError(f"end_sec ({end_sec}) must be greater than start_sec ({start_sec})")

    out_p.parent.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(str(in_p)) as f_in:
        sr = f_in.samplerate
        total_frames = len(f_in)
        channels = f_in.channels
        subtype = f_in.subtype

        start_frame = int(start_sec * sr)
        if start_frame >= total_frames:
            raise ValueError(
                f"start_sec ({start_sec}s) exceeds audio duration ({total_frames / sr:.2f}s)"
            )

        if end_sec is not None:
            end_frame = min(int(end_sec * sr), total_frames)
        else:
            end_frame = total_frames

        frames_to_read = end_frame - start_frame
        f_in.seek(start_frame)

        with sf.SoundFile(
            str(out_p),
            mode="w",
            samplerate=sr,
            channels=channels,
            subtype=subtype,
        ) as f_out:
            block_size = 65536
            remaining = frames_to_read
            while remaining > 0:
                chunk_size = min(block_size, remaining)
                data = f_in.read(chunk_size)
                if len(data) == 0:
                    break
                f_out.write(data)
                remaining -= len(data)

    return out_p
