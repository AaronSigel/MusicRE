from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import pytest
from typer.testing import CliRunner

from music_re.cli import app
from music_re.workspace import WorkspaceManager

runner = CliRunner()

@pytest.fixture
def mock_audio_workspace(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_wav = tmp_path / "test_track.wav"
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    sf.write(str(track_wav), y, sr)

    stems_dir = ws.get_stems_dir(track_wav, model_name="htdemucs")
    for s in ["vocals", "drums", "bass", "other"]:
        sf.write(str(stems_dir / f"{s}.wav"), y * 0.25, sr)
    (stems_dir / ".complete_htdemucs").write_text("ok")

    return ws, track_wav

def test_cli_evaluate_command(mock_audio_workspace):
    ws, track_wav = mock_audio_workspace
    with patch("music_re.cli.WorkspaceManager", return_value=ws):
        res = runner.invoke(app, ["evaluate", str(track_wav)])
        assert res.exit_code == 0
        assert "Quality Evaluation" in res.stdout

        res_json = runner.invoke(app, ["evaluate", str(track_wav), "--json"])
        assert res_json.exit_code == 0
        assert '"is_objective"' in res_json.stdout

def test_cli_export_command(tmp_path: Path, mock_audio_workspace):
    ws, track_wav = mock_audio_workspace
    out_dir = tmp_path / "final_export"

    with patch("music_re.cli.WorkspaceManager", return_value=ws):
        res = runner.invoke(app, ["export", str(track_wav), "--out", str(out_dir)])
        assert res.exit_code == 0
        assert "Export complete" in res.stdout
        assert out_dir.exists()
