#!/usr/bin/env python3
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict

import torch

def detect_hardware() -> Dict[str, Any]:
    rocm_hip = getattr(torch.version, "hip", None)
    cuda_avail = torch.cuda.is_available()
    rocm_avail = bool(rocm_hip and cuda_avail)

    cuda_device_name = None
    device_count = 0
    if cuda_avail:
        device_count = torch.cuda.device_count()
        if device_count > 0:
            cuda_device_name = torch.cuda.get_device_name(0)

    recommended_device = "cuda" if cuda_avail else "cpu"

    return {
        "cpu": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cores_logical": os.cpu_count() or 1,
            "threads": torch.get_num_threads(),
        },
        "cuda_available": cuda_avail and not rocm_avail,
        "rocm_available": rocm_avail,
        "rocm_hip_version": rocm_hip,
        "device_count": device_count,
        "device_name": cuda_device_name,
        "recommended_device": recommended_device,
        "torch_version": torch.__version__,
    }

def verify_model_loading(load_weights: bool = False) -> Dict[str, Any]:
    status = {}

    # 1. Demucs
    try:
        from demucs.pretrained import get_model
        if load_weights:
            model = get_model("htdemucs")
            status["demucs"] = {
                "status": "ready",
                "model_name": "htdemucs",
                "sources": model.sources,
                "samplerate": model.samplerate,
            }
        else:
            status["demucs"] = {
                "status": "metadata_ok",
                "default_model": "htdemucs",
            }
    except Exception as e:
        status["demucs"] = {
            "status": "error",
            "error": str(e),
        }

    # 2. Basic Pitch
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        onnx_model_path = Path(str(ICASSP_2022_MODEL_PATH) + ".onnx")
        exists = onnx_model_path.exists()
        if load_weights:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
            status["basic_pitch"] = {
                "status": "ready",
                "model_path": str(onnx_model_path),
                "providers": sess.get_providers(),
                "inputs": [i.name for i in sess.get_inputs()],
                "outputs": [o.name for o in sess.get_outputs()],
            }
        else:
            status["basic_pitch"] = {
                "status": "metadata_ok" if exists else "missing_file",
                "model_path": str(onnx_model_path),
                "exists": exists,
            }
    except Exception as e:
        status["basic_pitch"] = {
            "status": "error",
            "error": str(e),
        }

    return status

def main():
    print("=== MusicRE Hardware & Model Verification ===")
    hw = detect_hardware()
    print(f"Platform: {hw['cpu']['platform']}")
    print(f"Logical CPUs: {hw['cpu']['cores_logical']} (PyTorch threads: {hw['cpu']['threads']})")
    print(f"CUDA Available: {hw['cuda_available']} (Device count: {hw['device_count']})")
    if hw['device_name']:
        print(f"CUDA Device Name: {hw['device_name']}")
    print(f"ROCm Available: {hw['rocm_available']} (HIP: {hw['rocm_hip_version']})")
    print(f"Recommended Device: {hw['recommended_device']}")

    print("\n--- Verifying Model Definitions & Weights ---")
    models = verify_model_loading(load_weights=True)
    for name, info in models.items():
        print(f"[{name}] Status: {info.get('status')} | Details: {info}")

    report = {
        "hardware": hw,
        "models": models,
    }

    with open("hardware_verification.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nReport written to hardware_verification.json")

if __name__ == "__main__":
    main()
