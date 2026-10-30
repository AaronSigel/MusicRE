import pytest
import soundfile as sf
import numpy as np
from pathlib import Path
from unittest.mock import patch

from music_re.workspace import WorkspaceManager
from music_re.core.separator import separate_stems
from music_re.core.backends.separator_base import SeparatorBackend

class FaultySeparator(SeparatorBackend):
    backend_name = "faulty"
    model_name = "faulty_model"

    def load(self, device="cpu"):
        pass

    def unload(self):
        pass

    def supported_stems(self):
        return ["vocals", "drums"]
    
    def separate(self, input_audio, output_dir, shifts=1, overlap=0.25, progress_callback=None):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write first stem
        sf.write(str(out_dir / "vocals.wav"), np.zeros((22050, 2)), 22050)
        # Crash before completing second stem
        raise RuntimeError("Simulated crash during stem separation")

def test_failed_re_separation_leaves_previous_stems_intact(tmp_path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_wav = tmp_path / "song.wav"
    sf.write(str(track_wav), np.ones((22050, 2), dtype=np.float32) * 0.5, 22050)
    
    # 1. Establish valid previous stem cache
    stems_dir = ws.get_stems_dir(track_wav, model_name="faulty_model")
    stems_dir.mkdir(parents=True, exist_ok=True)
    orig_vocals = stems_dir / "vocals.wav"
    sf.write(str(orig_vocals), np.ones((22050, 2), dtype=np.float32) * 0.777, 22050)
    (stems_dir / ".complete_faulty_model").write_text("ok")
    
    ws.mark_stage_complete(track_wav, "separation", artifacts={"vocals": str(orig_vocals)})
    
    # 2. Run re-separation with failure
    backend = FaultySeparator("faulty", "faulty_model")
    with pytest.raises(RuntimeError, match="Simulated crash"):
        separate_stems(track_wav, workspace=ws, backend=backend, force=True)
        
    # 3. Assert previous stem set is still registered and content is unchanged
    manifest = ws.get_manifest(track_wav)
    assert "vocals" in manifest.stems
    data, _ = sf.read(str(orig_vocals))
    assert np.allclose(data, 0.777, atol=1e-4)

def test_empty_audio_raises_value_error(tmp_path):
    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    empty_wav = tmp_path / "empty.wav"
    sf.write(str(empty_wav), np.zeros((0, 2)), 22050)
    
    backend = FaultySeparator("faulty", "faulty_model")
    with pytest.raises(ValueError, match="empty|zero-length"):
        separate_stems(empty_wav, workspace=ws, backend=backend)
