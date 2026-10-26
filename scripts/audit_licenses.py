#!/usr/bin/env python3
import importlib.metadata
import json
import sys

PACKAGES = [
    "torch",
    "torchaudio",
    "demucs",
    "basic-pitch",
    "onnxruntime",
    "librosa",
    "pretty-midi",
    "music21",
    "pyloudnorm",
    "soundfile",
    "mutagen",
    "scipy",
    "numpy",
    "typer",
    "rich",
    "mcp",
    "pydantic",
]

MODEL_WEIGHT_LICENSES = {
    "demucs (htdemucs / mdx)": {
        "license": "CC-BY-NC 4.0 (Non-Commercial for official weights trained on Musdb-HQ)",
        "source": "https://github.com/facebookresearch/demucs#license",
        "commercial_use": False,
        "notes": "Code is MIT, but default pretrained weights are non-commercial due to Musdb-HQ license."
    },
    "basic-pitch (ICASSP 2022)": {
        "license": "Apache 2.0",
        "source": "https://github.com/spotify/basic-pitch",
        "commercial_use": True,
        "notes": "Code and provided ONNX weights are Apache 2.0."
    },
    "all-in-one-infer (candidate)": {
        "license": "MIT / Mixed dependencies",
        "source": "https://github.com/mir-aidj/all-in-one-infer",
        "commercial_use": "Conditional",
        "notes": "Pulls demucs-infer and madmom-infer. Check model checkpoint licenses (Harmonic-CNN, etc.)."
    }
}

def get_installed_license(pkg_name: str) -> str:
    try:
        meta = importlib.metadata.metadata(pkg_name)
        lic = meta.get("License")
        if not lic or lic.strip() == "UNKNOWN":
            classifiers = meta.get_all("Classifier") or []
            license_classifiers = [c.split("::")[-1].strip() for c in classifiers if "License ::" in c]
            if license_classifiers:
                lic = ", ".join(license_classifiers)
            else:
                lic = "Not specified in metadata"
        return lic.replace("\n", " ").strip()
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"

def main():
    report = {
        "packages": {},
        "models": MODEL_WEIGHT_LICENSES
    }
    print(f"{'Package':<20} | {'Version':<15} | {'License'}")
    print("-" * 75)
    for pkg in PACKAGES:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT INSTALLED"
        lic = get_installed_license(pkg)
        report["packages"][pkg] = {"version": ver, "license": lic}
        print(f"{pkg:<20} | {ver:<15} | {lic[:40]}")

    print("\n--- Model Weight Licenses ---")
    for model, info in MODEL_WEIGHT_LICENSES.items():
        print(f"[{model}]: {info['license']} (Commercial: {info['commercial_use']})")

    with open("license_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nFull license audit written to license_audit.json")

if __name__ == "__main__":
    main()
