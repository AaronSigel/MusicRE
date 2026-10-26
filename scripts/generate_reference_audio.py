#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import soundfile as sf

def generate_reference_audio(output_path: str | Path = "data/reference_10s.wav", duration_sec: float = 10.0, sr: int = 44100) -> Path:
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    # 1. Tonal melody (A4 440Hz + E5 659.25Hz alternating every 0.5s)
    melody = np.zeros_like(t)
    half_sec = int(sr * 0.5)
    for i in range(0, len(t), half_sec):
        freq = 440.0 if (i // half_sec) % 2 == 0 else 659.25
        seg_t = t[i : i + half_sec]
        melody[i : i + half_sec] = 0.3 * np.sin(2 * np.pi * freq * seg_t)

    # 2. Bass (A2 110Hz steady note)
    bass = 0.25 * np.sin(2 * np.pi * 110.0 * t)

    # 3. Drums: Kick (50Hz decaying pulse every 0.5s = 120 bpm quarter notes)
    kick = np.zeros_like(t)
    click_len = int(sr * 0.1)
    click_t = np.linspace(0, 0.1, click_len, endpoint=False)
    click_pulse = 0.5 * np.sin(2 * np.pi * 50.0 * click_t) * np.exp(-click_t * 30.0)
    for i in range(0, len(t), half_sec):
        end_idx = min(i + click_len, len(t))
        kick[i:end_idx] += click_pulse[: end_idx - i]

    # Combine stems into stereo mix
    left = melody + 0.8 * bass + kick
    right = 0.8 * melody + bass + kick

    stereo_mix = np.stack([left, right], axis=-1)
    # Normalize to -1.0 dB (~0.89 max)
    peak = np.max(np.abs(stereo_mix))
    if peak > 0:
        stereo_mix = (stereo_mix / peak) * 0.89

    sf.write(str(out_p), stereo_mix, sr, subtype="PCM_16")
    return out_p

if __name__ == "__main__":
    p = generate_reference_audio()
    print(f"Generated reference audio: {p}")
