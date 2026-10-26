from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from music_re.core.backends.demucs_backend import DemucsSeparator
from music_re.core.backends import get_separator_backend

def test_demucs_backend_metadata_four_stems():
    sep = DemucsSeparator(six_stems=False, device="cpu")
    assert sep.backend_name == "demucs"
    assert sep.model_name == "htdemucs"
    assert sep.supported_stems == ["vocals", "drums", "bass", "other"]
    assert sep.device == "cpu"

def test_demucs_backend_metadata_six_stems():
    sep = DemucsSeparator(six_stems=True, device="cpu")
    assert sep.model_name == "htdemucs_6s"
    assert sep.supported_stems == ["vocals", "drums", "bass", "other", "guitar", "piano"]
    assert hasattr(sep, "quality_note")
    assert "bleed" in sep.quality_note.lower()

@pytest.mark.models
def test_demucs_backend_load_and_unload():
    sep = DemucsSeparator(six_stems=False, device="cpu")
    assert sep._model is None
    sep.load()
    assert sep._model is not None
    model_ref = sep._model
    # Calling load again reuses same instance
    sep.load()
    assert sep._model is model_ref

    sep.unload()
    assert sep._model is None

def test_demucs_backend_separate_mock(tmp_path: Path):
    sr = 44100
    y = np.zeros((sr * 3 + 123, 2), dtype=np.float32)
    silent_wav = tmp_path / "silent.wav"
    sf.write(str(silent_wav), y, sr)

    out_dir = tmp_path / "stems"
    sep = DemucsSeparator(six_stems=False, device="cpu")
    results = sep.separate(silent_wav, out_dir)

    assert set(results.keys()) == set(sep.supported_stems)
    for stem_name, path in results.items():
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
        signal, rate = sf.read(path, always_2d=True)
        assert rate == sr
        assert signal.shape == (sr * 3 + 123, 2)
        assert not signal.any()
    assert sep._model is None
