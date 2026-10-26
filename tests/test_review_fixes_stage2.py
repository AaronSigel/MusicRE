from pathlib import Path
from unittest.mock import patch
import pytest

from music_re.core.backends.separator_base import resolve_device
from music_re.core.backends.demucs_backend import DemucsSeparator
from music_re.core.separator import separate_stems
from music_re.schemas import StemSeparationResult
from music_re.workspace import WorkspaceManager


def test_stem_separation_result_has_quality_note(tmp_path: Path):
    import numpy as np
    import soundfile as sf
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_file = tmp_path / "song.wav"
    sf.write(track_file, np.zeros(44100), 44100)

    res = separate_stems(track_file, workspace=ws, six_stems=True)
    assert isinstance(res, StemSeparationResult)
    assert res.quality_note is not None
    assert "bleed" in res.quality_note.lower()


def test_resolve_device_invalid_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(ValueError, match="CUDA device requested .* but CUDA is not available"):
            resolve_device("cuda")


def test_demucs_cleanup_on_exception(tmp_path: Path):
    track_file = tmp_path / "song.wav"
    track_file.write_bytes(b"RIFF\x00\x00\x00\x00")
    out_dir = tmp_path / "stems"

    sep = DemucsSeparator(six_stems=False, device="cpu")

    with patch("demucs.audio.AudioFile.read", side_effect=RuntimeError("Simulated decode explosion")):
        with pytest.raises(RuntimeError, match="Simulated decode explosion"):
            sep.separate(track_file, out_dir)

    # Verify no lingering tmp or complete markers
    assert not (out_dir / ".complete_htdemucs").exists()
    assert len(list(out_dir.glob("*_tmp.wav"))) == 0
