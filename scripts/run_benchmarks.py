#!/usr/bin/env python3
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import librosa
import pretty_midi
import soundfile as sf
import torch

from scripts.generate_reference_audio import generate_reference_audio


def benchmark_librosa(audio_path: str | Path) -> Dict[str, Any]:
    p = Path(audio_path)
    t0 = time.perf_counter()
    y, sr = librosa.load(str(p), sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, "__len__"):
        bpm_val = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        bpm_val = float(tempo)
    latency = time.perf_counter() - t0

    return {
        "component": "librosa",
        "duration_sec": float(librosa.get_duration(y=y, sr=sr)),
        "samplerate": int(sr),
        "bpm": round(bpm_val, 2),
        "beat_count": int(len(beat_frames)),
        "latency_sec": round(latency, 4),
    }


def benchmark_basic_pitch(audio_path: str | Path, tmp_dir: str | Path | None = None) -> Dict[str, Any]:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save

    in_p = Path(audio_path)

    def _run(work_dir: Path):
        t0 = time.perf_counter()
        onnx_model_path = Path(str(ICASSP_2022_MODEL_PATH) + ".onnx")
        model_to_use = str(onnx_model_path) if onnx_model_path.exists() else ICASSP_2022_MODEL_PATH

        with contextlib.redirect_stdout(sys.stderr):
            predict_and_save(
                audio_path_list=[str(in_p)],
                output_directory=str(work_dir),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=model_to_use,
            )
        latency = time.perf_counter() - t0
        midis = list(work_dir.glob("*.mid"))
        notes_count = 0
        if midis:
            pm = pretty_midi.PrettyMIDI(str(midis[0]))
            notes_count = sum(len(inst.notes) for inst in pm.instruments)

        return {
            "component": "basic_pitch",
            "latency_sec": round(latency, 4),
            "notes_count": notes_count,
            "midi_generated": len(midis) > 0,
        }

    if tmp_dir:
        return _run(Path(tmp_dir))
    with tempfile.TemporaryDirectory() as td:
        return _run(Path(td))


def benchmark_demucs(audio_path: str | Path, tmp_dir: str | Path | None = None) -> Dict[str, Any]:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    in_p = Path(audio_path)
    t0 = time.perf_counter()
    model = get_model("htdemucs")
    model.cpu()
    model.eval()

    # Load audio using torchaudio or soundfile
    data, sr = sf.read(str(in_p))
    if data.ndim == 1:
        data = data[:, None].repeat(2, axis=1)
    # Shape: (channels, samples)
    wav_tensor = torch.tensor(data.T, dtype=torch.float32)[None, ...]

    with torch.no_grad():
        sources = apply_model(model, wav_tensor, device="cpu", split=True)

    latency = time.perf_counter() - t0
    # sources shape: (1, num_sources, channels, samples)
    return {
        "component": "demucs",
        "model_name": "htdemucs",
        "source_names": model.sources,
        "sources_count": len(model.sources),
        "output_shape": list(sources.shape),
        "latency_sec": round(latency, 4),
    }


def main():
    print("=== Running MusicRE Feasibility Benchmarks ===")
    ref_audio = generate_reference_audio(output_path="data/reference_10s.wav", duration_sec=10.0)
    print(f"Prepared reference audio: {ref_audio}")

    results = {}

    tracemalloc.start()
    t_start = time.perf_counter()

    print("\n1. Benchmarking Librosa...")
    results["librosa"] = benchmark_librosa(ref_audio)
    print(f"   Done in {results['librosa']['latency_sec']}s (BPM: {results['librosa']['bpm']})")

    print("\n2. Benchmarking Basic Pitch (ONNX CPU)...")
    results["basic_pitch"] = benchmark_basic_pitch(ref_audio)
    print(f"   Done in {results['basic_pitch']['latency_sec']}s (Notes: {results['basic_pitch']['notes_count']})")

    print("\n3. Benchmarking Demucs (htdemucs CPU)...")
    results["demucs"] = benchmark_demucs(ref_audio)
    print(f"   Done in {results['demucs']['latency_sec']}s (Sources: {results['demucs']['source_names']})")

    total_time = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results["summary"] = {
        "audio_duration_sec": 10.0,
        "total_latency_sec": round(total_time, 2),
        "rtf": round(total_time / 10.0, 2),
        "peak_ram_mb": round(peak_mem / (1024 * 1024), 2),
    }

    print("\n--- Summary ---")
    print(f"Total processing time: {results['summary']['total_latency_sec']}s (RTF: {results['summary']['rtf']}x real-time)")
    print(f"Peak Python heap RAM: {results['summary']['peak_ram_mb']} MB")

    with open("benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nWrote results to benchmark_report.json")


if __name__ == "__main__":
    main()
