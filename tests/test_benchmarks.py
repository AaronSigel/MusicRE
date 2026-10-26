import sys
from pathlib import Path
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_reference_audio import generate_reference_audio
from scripts.run_benchmarks import benchmark_librosa, benchmark_basic_pitch

def test_generate_reference_audio(tmp_path):
    out_file = tmp_path / "ref_test.wav"
    path = generate_reference_audio(output_path=out_file, duration_sec=1.0, sr=22050)
    assert Path(path).exists()
    data, sr = sf.read(str(path))
    assert sr == 22050
    assert len(data.shape) in (1, 2)
    assert len(data) == 22050

def test_benchmark_librosa(tmp_path):
    out_file = tmp_path / "ref_test_librosa.wav"
    generate_reference_audio(output_path=out_file, duration_sec=1.5, sr=22050)
    res = benchmark_librosa(str(out_file))
    assert "latency_sec" in res
    assert "bpm" in res
    assert res["bpm"] > 0
    assert res["latency_sec"] > 0

def test_benchmark_basic_pitch(tmp_path):
    out_file = tmp_path / "ref_test_bp.wav"
    generate_reference_audio(output_path=out_file, duration_sec=1.0, sr=22050)
    res = benchmark_basic_pitch(str(out_file), tmp_dir=tmp_path)
    assert "latency_sec" in res
    assert "notes_count" in res
    assert res["latency_sec"] > 0
