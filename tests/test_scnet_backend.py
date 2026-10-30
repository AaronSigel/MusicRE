from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import soundfile as sf
import torch
from typer.testing import CliRunner

from music_re.cli import app
from music_re.core.backends import get_separator_backend, list_separator_backends
from music_re.core.backends.scnet_backend import SCNet, SCNetSeparator
from music_re.core.separator import separate_stems
from music_re.schemas import StemSeparationResult
from music_re.workspace import WorkspaceManager


def test_scnet_registration():
    backends = list_separator_backends()
    assert "demucs" in backends
    assert "scnet" in backends

    backend = get_separator_backend("scnet")
    assert isinstance(backend, SCNetSeparator)
    assert backend.backend_name == "scnet"
    assert backend.model_name == "scnet"
    assert backend.supported_stems == ["drums", "bass", "other", "vocals"]


def test_scnet_device_and_load_unload():
    separator = SCNetSeparator(device="cpu")
    assert separator.device == "cpu"
    assert separator._model is None

    separator.load()
    assert separator._model is not None
    assert isinstance(separator._model, SCNet)

    separator.unload()
    assert separator._model is None


def test_scnet_checkpoint_loading(tmp_path: Path):
    model = SCNet()
    ckpt_path = tmp_path / "scnet_dummy.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    separator = SCNetSeparator(checkpoint_path=ckpt_path, device="cpu")
    separator.load()
    assert separator._model is not None
    separator.unload()

    missing_separator = SCNetSeparator(checkpoint_path=tmp_path / "nonexistent.pt", device="cpu")
    with pytest.raises(FileNotFoundError):
        missing_separator.load()


def test_scnet_separate_mocked_model(tmp_path: Path):
    audio_file = tmp_path / "test.wav"
    sr = 44100
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32)
    stereo = np.column_stack([y, y])
    sf.write(str(audio_file), stereo, sr)

    out_dir = tmp_path / "stems_out"
    separator = SCNetSeparator(device="cpu")

    # Mock forward pass to return predictable stereo tensors
    mock_model = MagicMock()
    mock_model.num_sources = 4
    # Shape: [1, 4, 2, sr]
    dummy_sources = torch.zeros(1, 4, 2, sr, dtype=torch.float32)
    mock_model.return_value = dummy_sources
    separator._model = mock_model

    result = separator.separate(audio_file, out_dir, shifts=1, overlap=0.25)

    assert set(result.keys()) == {"drums", "bass", "other", "vocals"}
    for stem_name, path in result.items():
        assert path.is_file()
        assert path.name == f"{stem_name}.wav"
        data, file_sr = sf.read(str(path))
        assert file_sr == 44100
        assert data.shape == (sr, 2)


def test_scnet_separate_silent_audio(tmp_path: Path):
    audio_file = tmp_path / "silent.wav"
    sr = 44100
    silent = np.zeros((sr, 2), dtype=np.float32)
    sf.write(str(audio_file), silent, sr)

    out_dir = tmp_path / "stems_silent"
    separator = SCNetSeparator(device="cpu")
    result = separator.separate(audio_file, out_dir)

    assert set(result.keys()) == {"drums", "bass", "other", "vocals"}
    for path in result.values():
        assert path.is_file()
        data, file_sr = sf.read(str(path))
        assert file_sr == 44100
        assert np.all(data == 0)


def test_separate_stems_with_scnet_backend(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_file = tmp_path / "track.wav"
    sr = 44100
    # 0.5s audio
    sig = (0.5 * np.sin(np.linspace(0, 100, sr // 2))).astype(np.float32)
    sf.write(str(track_file), np.column_stack([sig, sig]), sr)

    res = separate_stems(track_file, workspace=ws, backend_name="scnet", device="cpu")
    assert isinstance(res, StemSeparationResult)
    assert res.model_used == "scnet"
    assert set(res.stems.keys()) == {"drums", "bass", "other", "vocals"}
    for path_str in res.stems.values():
        p = Path(path_str)
        assert p.is_file()
        assert p.stat().st_size > 0


def test_cli_split_and_deconstruct_options():
    runner = CliRunner()
    split_res = runner.invoke(app, ["split", "--help"])
    assert split_res.exit_code == 0
    assert "--backend" in split_res.output

    decon_res = runner.invoke(app, ["deconstruct", "--help"])
    assert decon_res.exit_code == 0
    assert "--separator-backend" in decon_res.output
