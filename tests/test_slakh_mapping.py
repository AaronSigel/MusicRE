import sys
from pathlib import Path

import pretty_midi
import pytest
import soundfile as sf
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.slakh_mapping import classify_stem, estimate_octave_shift, prepare_track

SR = 22050


def test_estimate_octave_shift_detects_audio_one_octave_below_midi():
    import numpy as np
    notes = [pretty_midi.Note(velocity=100, pitch=48, start=i, end=i + .6) for i in range(8)]
    t = np.arange(SR * 9) / SR
    audio = sum((.2 / harmonic) * np.sin(2 * np.pi * 65.406 * harmonic * t)
                for harmonic in range(1, 4))
    shift, diagnostics = estimate_octave_shift(audio, SR, notes)
    assert shift == -12
    assert diagnostics['measured'] >= 8 and diagnostics['agreement'] >= .6


def test_estimate_octave_shift_returns_zero_for_matching_noise_and_few_notes():
    import numpy as np
    notes = [pretty_midi.Note(velocity=100, pitch=48, start=i, end=i + .5) for i in range(8)]
    t = np.arange(SR * 9) / SR
    audio = sum((.2 / h) * np.sin(2 * np.pi * 130.813 * h * t) for h in range(1, 4))
    assert estimate_octave_shift(audio, SR, notes)[0] == 0
    assert estimate_octave_shift(np.random.default_rng(1).normal(0, .1, SR * 9), SR, notes)[0] == 0
    shift, diagnostics = estimate_octave_shift(audio, SR, notes[:2])
    assert shift == 0 and diagnostics['measured'] < 8


def _write_fake_slakh_track(track_dir: Path, stems, duration_sec: float = 1.0):
    """Mirror the ethman/slakh-utils layout closely enough for prepare_track."""
    track_dir.mkdir(parents=True, exist_ok=True)
    (track_dir / "MIDI").mkdir(exist_ok=True)
    (track_dir / "stems").mkdir(exist_ok=True)

    n_samples = int(SR * duration_sec)
    mix = [0.0] * n_samples
    metadata = {"stems": {}}

    for i, (stem_id, inst_class, is_drum, program_num) in enumerate(stems):
        metadata["stems"][stem_id] = dict(
            inst_class=inst_class,
            is_drum=is_drum,
            midi_program_name=inst_class,
            program_num=program_num,
            audio_rendered=True,
        )

        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=program_num, is_drum=is_drum)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36 if is_drum else 60,
                                            start=0.1 * (i + 1), end=0.1 * (i + 1) + 0.2))
        pm.instruments.append(inst)
        pm.write(str(track_dir / "MIDI" / f"{stem_id}.mid"))

        import numpy as np
        freq = 220.0 * (i + 1)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        wave = (0.1 * np.sin(2 * np.pi * freq * t)).tolist()
        sf.write(str(track_dir / "stems" / f"{stem_id}.wav"), wave, SR)
        mix = [m + w for m, w in zip(mix, wave)]

    (track_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    sf.write(str(track_dir / "mix.wav"), mix, SR)
    return track_dir


def _desync_one_stem_wav(track_dir: Path, stem_id: str, extra_silence_sec: float):
    import numpy as np
    path = track_dir / "stems" / f"{stem_id}.wav"
    y, sr = sf.read(str(path))
    padded = np.concatenate([y, [0.0] * int(sr * extra_silence_sec)])
    sf.write(str(path), padded, sr)


def test_classify_stem_maps_known_and_unknown_classes():
    assert classify_stem("Drums", is_drum=True) == "drums"
    assert classify_stem("Bass", is_drum=False) == "bass"  # exact Slakh inst_class; "Electric Bass" is a midi_program_name
    assert classify_stem("Theremin", is_drum=False) == "other"  # unmapped class, Review Focus #2


def test_prepare_track_builds_grouped_ground_truth(tmp_path):
    track_dir = _write_fake_slakh_track(tmp_path / "Track00001", stems=[
        ("S00", "Guitar", False, 48),
        ("S01", "Drums", True, 0),
        ("S02", "Bass", False, 33),
    ])  # no vocal stem at all
    out_dir = tmp_path / "prepared"
    result = prepare_track(track_dir, out_dir)

    assert result["track_id"] == "Track00001"
    assert result["octave_shift"] == {"vocals": 0, "drums": 0, "bass": 0, "other": 0}
    gt_dir = Path(result["ground_truth_stems_dir"])
    assert (gt_dir / "drums.wav").exists()
    assert (gt_dir / "bass.wav").exists()
    assert (gt_dir / "other.wav").exists()
    assert not (gt_dir / "vocals.wav").exists()  # Review Focus #1

    gt_pm = pretty_midi.PrettyMIDI(result["ground_truth_midi_path"])
    drum_insts = [i for i in gt_pm.instruments if i.is_drum]
    pitched_insts = [i for i in gt_pm.instruments if not i.is_drum]
    assert sum(len(i.notes) for i in drum_insts) == 1
    assert sum(len(i.notes) for i in pitched_insts) == 2

    # Review C2: Demucs always outputs 2-channel stems (channels=2,
    # hardcoded in demucs_backend.py); a mono ground-truth reference fails
    # separation_quality's shape check and SDR is silently never computed.
    drums_info = sf.info(str(gt_dir / "drums.wav"))
    assert drums_info.channels == 2


def test_prepare_track_transposes_only_bass_reference(tmp_path, monkeypatch):
    track_dir = _write_fake_slakh_track(tmp_path / "Track00021", stems=[
        ("S00", "Guitar", False, 48), ("S01", "Bass", False, 33),
    ])
    monkeypatch.setattr("scripts.slakh_mapping.estimate_octave_shift", lambda *a: (-12, {"agreement": 1.0}))
    result = prepare_track(track_dir, tmp_path / "prepared-shift")
    refs = pretty_midi.PrettyMIDI(result["ground_truth_midi_path"])
    notes = {inst.name: [n.pitch for n in inst.notes] for inst in refs.instruments}
    assert notes == {"bass": [48], "other": [60]}
    assert result["octave_shift"]["bass"] == -12
    assert any("octave mismatch" in w for w in result["warnings"])


def test_prepare_track_handles_missing_stem_audio_without_crashing(tmp_path):
    # Review I5: Slakh's metadata.yaml "audio_rendered" flag is unreliable
    # in practice (BabySlakh Track00020 marks every stem False yet ships
    # the wavs); existence on disk is the safer test, and a missing wav
    # must not crash the whole track -- the MIDI still describes what the
    # composition contains, so its notes stay in the ground truth.
    track_dir = _write_fake_slakh_track(tmp_path / "Track00003", stems=[
        ("S00", "Guitar", False, 48),
        ("S01", "Drums", True, 0),
    ])
    (track_dir / "stems" / "S01.wav").unlink()

    result = prepare_track(track_dir, tmp_path / "prepared3")

    assert result["group_note_counts"]["drums"] == 1
    assert not (Path(result["ground_truth_stems_dir"]) / "drums.wav").exists()
    assert any("S01" in w for w in result["warnings"])


def test_prepare_track_rejects_misaligned_ground_truth(tmp_path):
    track_dir = _write_fake_slakh_track(tmp_path / "Track00002", stems=[("S00", "Guitar", False, 48)])
    _desync_one_stem_wav(track_dir, "S00", extra_silence_sec=5.0)  # corrupt fixture, Review Focus-adjacent

    with pytest.raises(ValueError, match="alignment"):
        prepare_track(track_dir, tmp_path / "prepared2")
