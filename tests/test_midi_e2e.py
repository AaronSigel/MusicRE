from pathlib import Path
import numpy as np
import pretty_midi
import soundfile as sf
import pytest

from music_re.core.backends import get_transcriber_backend
from music_re.core.midi_project import build_combined_midi_project
from music_re.schemas import MusicalTimeline, TimelineSection
from music_re.workspace import WorkspaceManager


@pytest.fixture
def test_audio_stems(tmp_path: Path):
    sr = 22050
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    # 1. Melodic vocals tone (A4 440 Hz)
    y_vocals = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    # 2. Bass tone (A2 110 Hz)
    y_bass = (0.5 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
    # 3. Drums pulse at 2 Hz
    y_drums = (0.5 * np.sin(2 * np.pi * 2.0 * t) * (np.sin(2 * np.pi * 2.0 * t) > 0.9)).astype(np.float32)

    stems = {
        "vocals": tmp_path / "vocals.wav",
        "bass": tmp_path / "bass.wav",
        "drums": tmp_path / "drums.wav",
    }
    sf.write(str(stems["vocals"]), y_vocals, sr)
    sf.write(str(stems["bass"]), y_bass, sr)
    sf.write(str(stems["drums"]), y_drums, sr)

    timeline = MusicalTimeline(
        duration_sec=duration_sec,
        bpm=120.0,
        beats=[0.0, 0.5, 1.0, 1.5],
        downbeats=[0.0],
        sections=[TimelineSection(label="intro", start_sec=0.0, end_sec=2.0, confidence=1.0)],
    )

    return stems, timeline


def test_stage4_e2e_transcription_and_project_assembly(tmp_path: Path, test_audio_stems):
    stems, timeline = test_audio_stems
    midis_dir = tmp_path / "midis"
    midis_dir.mkdir(parents=True, exist_ok=True)

    stem_midis = {}
    for name, audio_p in stems.items():
        out_mid = midis_dir / f"{name}.mid"
        mode = "drums" if name == "drums" else "tonal"
        transcriber = get_transcriber_backend(mode=mode)
        transcriber.transcribe(
            audio_path=audio_p,
            output_midi_path=out_mid,
            timeline=timeline,
            bpm=120.0,
            quantize=True,
            save_raw=True,
        )

        assert out_mid.exists()
        raw_mid = midis_dir / f"{name}_raw.mid"
        assert raw_mid.exists(), f"Raw MIDI missing for stem {name}"
        stem_midis[name] = out_mid

        # Verify no negative durations or hanging notes
        pm = pretty_midi.PrettyMIDI(str(out_mid))
        for inst in pm.instruments:
            for n in inst.notes:
                assert n.end > n.start, f"Invalid note duration: start={n.start}, end={n.end}"
                assert n.start >= 0.0

    # Build combined all_stems project
    all_stems_file = midis_dir / "all_stems.mid"
    build_combined_midi_project(stem_midis, all_stems_file, timeline=timeline)
    assert all_stems_file.exists()

    combined_pm = pretty_midi.PrettyMIDI(str(all_stems_file))
    assert len(combined_pm.instruments) == 3
    names = {inst.name.lower() for inst in combined_pm.instruments}
    assert names == {"vocals", "bass", "drums"}

    # Verify channel/is_drum flags
    for inst in combined_pm.instruments:
        if inst.name.lower() == "drums":
            assert inst.is_drum is True
        else:
            assert inst.is_drum is False
