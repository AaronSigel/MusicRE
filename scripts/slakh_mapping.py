#!/usr/bin/env python3
"""Map a raw Slakh2100/BabySlakh track to MusicRE's four stem groups and build
an aligned, grouped ground truth (stems + one combined MIDI) for benchmarking.

Track layout expected (https://github.com/ethman/slakh-utils#at-a-glance):
    TrackXXXXX/metadata.yaml, TrackXXXXX/MIDI/SXX.mid, TrackXXXXX/stems/SXX.wav,
    TrackXXXXX/mix.wav
"""
import os
import shutil
import sys
import tempfile
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

# Exact Slakh2100 inst_class values (slakh-utils' taxonomy). Matching is
# exact, never by substring: "Bassoon" is a Reed patch, not a bass, and
# Slakh has no vocal class at all ("Choir Aahs" is a Strings patch).
# Patch names such as "Electric Bass (finger)" live in midi_program_name,
# not inst_class. Any class not listed here maps to "other".
GROUP_BY_INST_CLASS: Dict[str, str] = {
    "Drums": "drums",
    "Bass": "bass",
}

KNOWN_INST_CLASSES = frozenset({
    "Piano", "Chromatic Percussion", "Organ", "Guitar", "Bass", "Strings",
    "Strings (continued)", "Brass", "Reed", "Pipe", "Synth Lead", "Synth Pad",
    "Sound Effects", "Ethnic", "Percussive", "Drums",
})

GROUPS = ("vocals", "drums", "bass", "other")

# Slakh stems and their mix are rendered from the same MIDI at the same
# sample rate; any larger gap is a real desync, not rounding.
ALIGNMENT_TOLERANCE_SEC = 0.05
ALLOWED_OCTAVE_SHIFTS = frozenset({-24, -12, 0, 12})


def estimate_octave_shift(audio: np.ndarray, sr: int, notes: list[pretty_midi.Note]) -> tuple[int, dict]:
    """Estimate robust MIDI-to-audio octave offset from voiced note fundamentals."""
    import librosa
    duration = len(audio) / sr if sr else 0
    windows = [(min(max(0.0, duration - 10.0), i * max(0.0, duration - 10.0) / 5),
                min(duration, 10.0 + i * max(0.0, duration - 10.0) / 5)) for i in range(6)]
    windows = list(dict.fromkeys((a, b) for a, b in windows if b > a))
    histogram: Dict[int, int] = {}
    measured = 0
    visited = set()
    for start, end in windows:
        segment = np.asarray(audio[int(start * sr):int(end * sr)], dtype=float)
        if len(segment) < 4096:
            continue
        f0, voiced, voiced_prob = librosa.pyin(segment, fmin=25, fmax=500, sr=sr,
                                               frame_length=4096, hop_length=256)
        times = librosa.times_like(f0, sr=sr, hop_length=256)
        for note_index, note in enumerate(notes):
            if note_index in visited:
                continue
            onset = max(note.start, start)
            offset = min(note.end, end)
            trim_start = max(onset + min(.05, .3 * (note.end - note.start)), start)
            if offset <= trim_start:
                continue
            mask = ((times >= trim_start - start) & (times <= offset - start) & voiced
                    & (voiced_prob >= .1) & np.isfinite(f0))
            values = f0[mask]
            if len(values) < 3:
                continue
            median_midi = 69 + 12 * np.log2(np.median(values) / 440.0)
            shift = int(round((median_midi - note.pitch) / 12) * 12)
            histogram[shift] = histogram.get(shift, 0) + 1
            measured += 1
            visited.add(note_index)
    agreement = max(histogram.values(), default=0) / measured if measured else 0.0
    shift = max(histogram, key=histogram.get) if measured >= 8 and agreement >= .6 else 0
    return shift, dict(measured=measured, agreement=agreement, histogram=histogram,
                       windows=[(round(a, 3), round(b, 3)) for a, b in windows])

# Per-group reference completeness. Only ABSENT means "this instrument is not
# in the song"; the other non-present states mean the reference itself is
# incomplete, so notes the model finds there are not false positives.
PRESENT = "present"                        # every stem has audio and MIDI
ABSENT = "absent"                          # no stem of this group in metadata
MISSING_ANNOTATION = "missing_annotation"  # some stem has audio but no MIDI
NOT_RENDERED = "not_rendered"              # some stem has MIDI but no audio


def classify_stem(inst_class: str, is_drum: bool) -> str:
    """One of MusicRE's four groups for a single Slakh stem."""
    if is_drum:
        return "drums"
    return GROUP_BY_INST_CLASS.get(inst_class, "other")


def _group_status(stem_files: list) -> str:
    """Status of one group from its stems' (has_audio, has_midi) pairs.

    Stems with neither file are filtered out before this is called.
    """
    if not stem_files:
        return ABSENT
    if any(audio and not midi for audio, midi in stem_files):
        return MISSING_ANNOTATION
    if any(midi and not audio for audio, midi in stem_files):
        return NOT_RENDERED
    return PRESENT


def _publish_dir(staged: Path, out_dir: Path, staging_root: Path) -> None:
    """Atomically make staged the new out_dir, restoring the old one on failure."""
    if not out_dir.exists():
        os.replace(staged, out_dir)
        return
    previous = staging_root / "previous"
    os.replace(out_dir, previous)
    try:
        os.replace(staged, out_dir)
    except BaseException:
        os.replace(previous, out_dir)
        raise


def prepare_track(track_dir: Path, out_dir: Path) -> dict:
    """Build grouped ground-truth stems and one combined ground-truth MIDI.

    Returns a dict with track_id, mix_path (the original, untouched mix),
    ground_truth_stems_dir, ground_truth_midi_path, group_note_counts
    (0 for a group with no Slakh stem mapped to it), group_status (one of
    present / absent / missing_annotation / not_rendered per group, see
    _group_status) and warnings.

    out_dir is built in a fresh staging dir next to it and swapped in as a
    whole, so a re-run never leaves files from an earlier run behind and a
    failed run leaves the previous out_dir untouched.
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
    group_files: Dict[str, list] = {g: [] for g in GROUPS}
    octave_shift_by_source: Dict[str, dict] = {g: {} for g in GROUPS}
    warnings: list = []

    for stem_id, info in metadata["stems"].items():
        inst_class = info["inst_class"]
        if inst_class not in KNOWN_INST_CLASSES and not info.get("is_drum", False):
            warnings.append(f"{stem_id}: unknown inst_class {inst_class!r}, mapped to 'other'")
        group = classify_stem(inst_class, bool(info.get("is_drum", False)))

        # metadata.yaml's audio_rendered / midi_saved flags are unreliable in
        # practice (BabySlakh marks every stem False yet ships the files);
        # existence on disk is the only check we trust.
        stem_wav = track_dir / "stems" / f"{stem_id}.wav"
        stem_midi_path = track_dir / "MIDI" / f"{stem_id}.mid"
        has_audio, has_midi = stem_wav.is_file(), stem_midi_path.is_file()
        if not has_audio and not has_midi:
            # Real BabySlakh lists such stems (e.g. Track00001/S06): they are
            # in neither the mix nor the annotation, so they say nothing
            # about the reference's completeness.
            warnings.append(f"{stem_id} ({group}): listed in metadata but has neither audio nor MIDI, excluded")
            continue
        group_files[group].append((has_audio, has_midi))

        y_mono = None
        if has_audio:
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
        else:
            # A stem with no audio still has notes describing the
            # composition, so skip only the audio contribution.
            warnings.append(f"{stem_id} ({group}): audio file missing, kept MIDI notes without audio contribution")

        if has_midi:
            pm = pretty_midi.PrettyMIDI(str(stem_midi_path))
            source_notes = [note for inst in pm.instruments for note in inst.notes]
            if group == "bass":
                shift = 0
                diagnostics = {"measured": 0, "agreement": 0.0, "histogram": {}}
                if y_mono is not None and source_notes:
                    shift, diagnostics = estimate_octave_shift(y_mono, sr, source_notes)
                if shift not in ALLOWED_OCTAVE_SHIFTS:
                    warnings.append(f"{stem_id} (bass): implausible octave shift {shift:+d} semitones; ignored")
                    shift = 0
                octave_shift_by_source[group][stem_id] = shift
                if shift:
                    for note in source_notes:
                        note.pitch = int(np.clip(note.pitch + shift, 0, 127))
                    warnings.append(f"{stem_id} (bass): audio/MIDI octave mismatch detected ({shift:+d} semitones; "
                                    f"{diagnostics['agreement']:.0%} agreement)")
            for inst in pm.instruments:
                group_notes[group].extend(inst.notes)
                group_note_counts[group] += len(inst.notes)
        else:
            warnings.append(f"{stem_id} ({group}): MIDI file missing, group reference annotation is incomplete")

    group_status = {g: _group_status(group_files[g]) for g in GROUPS}
    octave_shift = {g: 0 for g in GROUPS}
    if octave_shift_by_source["bass"]:
        counts = {}
        for shift in octave_shift_by_source["bass"].values():
            counts[shift] = counts.get(shift, 0) + 1
        octave_shift["bass"] = max(counts, key=counts.get)

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

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{out_dir.name}.staging-"))
    try:
        staged = staging_root / "track"
        staged_stems = staged / "ground_truth_stems"
        staged_stems.mkdir(parents=True)
        for group, audio in group_audio.items():
            # Demucs always outputs 2-channel stems (channels=2 in
            # demucs_backend.py); duplicate mono Slakh audio to stereo so
            # separation_quality's shape check matches and SDR gets computed.
            # FLOAT keeps a group sum above 1.0 instead of clipping it.
            sf.write(str(staged_stems / f"{group}.wav"), np.stack([audio, audio], axis=-1), sr,
                     subtype="FLOAT")
        gt_pm.write(str(staged / "ground_truth.mid"))
        _publish_dir(staged, out_dir, staging_root)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return {
        "track_id": track_id,
        "mix_path": str(mix_path),
        "ground_truth_stems_dir": str(out_dir / "ground_truth_stems"),
        "ground_truth_midi_path": str(out_dir / "ground_truth.mid"),
        "group_note_counts": group_note_counts,
        "group_status": group_status,
        "octave_shift": octave_shift,
        "octave_shift_by_source": octave_shift_by_source,
        "warnings": warnings,
    }
