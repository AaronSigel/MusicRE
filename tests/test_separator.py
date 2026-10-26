from pathlib import Path
from unittest.mock import patch
import pytest
from music_re.core.separator import separate_stems
from music_re.schemas import StemSeparationResult
from music_re.workspace import WorkspaceManager

def test_separate_stems_uses_cache(tmp_path: Path, seed_stem_cache):
    import numpy as np
    import soundfile as sf
    ws = WorkspaceManager(cache_root=tmp_path / ".cache")
    track_file = tmp_path / "song.wav"
    sf.write(track_file, np.zeros((44100, 2)), 44100)

    track_dir = ws.get_track_dir(track_file)
    stems_dir = track_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    stems = {name: stems_dir / f'{name}.wav' for name in ('vocals', 'drums', 'bass', 'other')}
    for path in stems.values():
        sf.write(path, np.zeros((44100, 2)), 44100)
    seed_stem_cache(track_file, stems)
    with patch('music_re.core.backends.demucs_backend.DemucsSeparator.separate', side_effect=AssertionError('must reuse valid cache')):
        result = separate_stems(track_file, workspace=ws, six_stems=False, device='cpu')
    assert isinstance(result, StemSeparationResult)
    assert result.model_used == "htdemucs"
    assert "vocals" in result.stems
    assert "drums" in result.stems
    assert "bass" in result.stems
    assert "other" in result.stems
    assert Path(result.stems["vocals"]).exists()

def test_separate_stems_file_not_found(tmp_path: Path):
    ws = WorkspaceManager(cache_root=tmp_path / ".cache")
    with pytest.raises(FileNotFoundError):
        separate_stems(tmp_path / "missing.wav", workspace=ws)
