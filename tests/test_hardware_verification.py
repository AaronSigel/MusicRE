import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.verify_hardware_and_models import detect_hardware, verify_model_loading

def test_detect_hardware():
    hw = detect_hardware()
    assert "cpu" in hw
    assert "cuda_available" in hw
    assert "rocm_available" in hw
    assert "device_count" in hw
    assert "recommended_device" in hw
    assert hw["cpu"]["threads"] > 0

def test_verify_model_loading_metadata():
    status = verify_model_loading(load_weights=False)
    assert "basic_pitch" in status
    assert "demucs" in status
    assert status["basic_pitch"]["status"] in ("ready", "metadata_ok")
    assert status["demucs"]["status"] in ("ready", "metadata_ok")


def test_onnx_probe_does_not_depend_on_default_runtime(monkeypatch):
    import basic_pitch
    monkeypatch.setattr(basic_pitch, 'ICASSP_2022_MODEL_PATH', basic_pitch.build_icassp_2022_model_path(basic_pitch.FilenameSuffix.onnx))
    status = verify_model_loading(load_weights=False)
    assert status['basic_pitch']['status'] == 'metadata_ok'
