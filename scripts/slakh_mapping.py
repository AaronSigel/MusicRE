#!/usr/bin/env python3
"""Map a raw Slakh2100/BabySlakh track to MusicRE's four stem groups and build
an aligned, grouped ground truth (stems + one combined MIDI) for benchmarking.

Track layout expected (https://github.com/ethman/slakh-utils#at-a-glance):
    TrackXXXXX/metadata.yaml, TrackXXXXX/MIDI/SXX.mid, TrackXXXXX/stems/SXX.wav,
    TrackXXXXX/mix.wav
"""
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pretty_midi
import soundfile as sf
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from music_re.core.audio_io import decode_audio

# Slakh's own (canonical) inst_class strings that map onto MusicRE's non-drum,
# non-bass, non-vocal group are everything GROUP_BY_INST_CLASS.get() misses,
# so this table only needs the classes that are NOT "other". "vocal" and
# "bass" are matched by substring in classify_stem(), because Slakh data (and
# test fixtures) name specific patches ("Electric Bass") rather than always
# using the bare class name ("Bass"); an exact-match-only table would silently
# drop those into "other".
GROUP_BY_INST_CLASS: Dict[str, str] = {
    "Drums": "drums",
    "Bass": "bass",
}

GROUPS = ("vocals", "drums", "bass", "other")

# Slakh stems and their mix are rendered from the same MIDI at the same
# sample rate; any larger gap is a real desync, not rounding.
ALIGNMENT_TOLERANCE_SEC = 0.05


def classify_stem(inst_class: str, is_drum: bool) -> str:
    """One of MusicRE's four groups for a single Slakh stem."""
    if is_drum:
        return "drums"
    lowered = inst_class.lower()
    if "vocal" in lowered:
        return "vocals"
    if "bass" in lowered:
        return "bass"
    return GROUP_BY_INST_CLASS.get(inst_class, "other")


def prepare_track(track_dir: Path, out_dir: Path) -> dict:
    """Build grouped ground-truth stems and one combined ground-truth MIDI.

    Returns a dict with track_id, mix_path (the original, untouched mix),
    ground_truth_stems_dir, ground_truth_midi_path, and group_note_counts
    (0 for a group with no Slakh stem mapped to it).
    """
    track_dir = Path(track_dir).resolve()
    out_dir = Path(out_dir).resolve()
    track_id = track_dir.name

    metadata = yaml.safe_load((track_dir / "metadata.yaml").read_text())
    mix_path = track_dir / "mix.wav"
    mix, sr = decode_audio(mix_path, target_sr=None, normalize_db=None)
    mix_mono = mix if mix.ndim == 1 else mix.mean(axis=1)
    n_samples = len(mix_mono)
    mix_duration_sec = n_samples / sr
    tolerance_samples = int(ALIGNMENT_TOLERANCE_SEC * sr)

    group_audio: Dict[str, np.ndarray] = {}
    group_notes: Dict[str, list] = {g: [] for g in GROUPS}
    group_note_counts: Dict[str, int] = {g: 0 for g in GROUPS}

    for stem_id, info in metadata["stems"].items():
        group = classify_stem(info["inst_class"], bool(info.get("is_drum", False)))

        stem_wav = track_dir / "stems" / f"{stem_id}.wav"
        y, _ = decode_audio(stem_wav, target_sr=sr, normalize_db=None)
        y_mono = y if y.ndim == 1 else y.mean(axis=1)

        diff = len(y_mono) - n_samples
        if abs(diff) > tolerance_samples:
            raise ValueError(
                f"{track_id}/{stem_id}: audio/MIDI alignment check failed "
                f"({len(y_mono)} samples vs mix's {n_samples} samples, "
                f"tolerance {tolerance_samples} samples)"
            )
        if diff < 0:
            y_mono = np.pad(y_mono, (0, -diff))
        elif diff > 0:
            y_mono = y_mono[:n_samples]

        buffer = group_audio.get(group)
        group_audio[group] = y_mono.copy() if buffer is None else buffer + y_mono

        stem_midi_path = track_dir / "MIDI" / f"{stem_id}.mid"
        if stem_midi_path.exists():
            pm = pretty_midi.PrettyMIDI(str(stem_midi_path))
            for inst in pm.instruments:
                group_notes[group].extend(inst.notes)
                group_note_counts[group] += len(inst.notes)

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_stems_dir = out_dir / "ground_truth_stems"
    gt_stems_dir.mkdir(parents=True, exist_ok=True)
    for group, audio in group_audio.items():
        sf.write(str(gt_stems_dir / f"{group}.wav"), audio, sr)

    gt_pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    max_note_end = 0.0
    for group in GROUPS:
        notes = group_notes[group]
        if not notes:
            continue
        inst = pretty_midi.Instrument(program=0, is_drum=(group == "drums"), name=group)
        inst.notes.extend(notes)
        gt_pm.instruments.append(inst)
        max_note_end = max(max_note_end, max(n.end for n in notes))

    if gt_pm.instruments and max_note_end - mix_duration_sec > ALIGNMENT_TOLERANCE_SEC:
        raise ValueError(
            f"{track_id}: ground truth MIDI/audio alignment check failed "
            f"(last note ends at {max_note_end:.3f}s, mix duration {mix_duration_sec:.3f}s)"
        )

    gt_midi_path = out_dir / "ground_truth.mid"
    gt_pm.write(str(gt_midi_path))

    return {
        "track_id": track_id,
        "mix_path": str(mix_path),
        "ground_truth_stems_dir": str(gt_stems_dir),
        "ground_truth_midi_path": str(gt_midi_path),
        "group_note_counts": group_note_counts,
    }
