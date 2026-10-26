from pathlib import Path
import pretty_midi
import pytest

from music_re.core.midi_project import build_combined_midi_project
from music_re.schemas import MusicalTimeline

@pytest.fixture
def sample_stem_midis(tmp_path: Path):
    midis = {}
    stems = ["vocals", "drums", "bass", "other"]
    for s in stems:
        p = tmp_path / f"{s}.mid"
        pm = pretty_midi.PrettyMIDI(initial_tempo=128.0)
        is_drum = (s == "drums")
        inst = pretty_midi.Instrument(program=0, is_drum=is_drum, name=s.capitalize())
        pitch = 36 if is_drum else (40 if s == "bass" else 60)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=0.0, end=0.5))
        pm.instruments.append(inst)
        pm.write(str(p))
        midis[s] = p
    return midis

def test_build_combined_midi_project(tmp_path: Path, sample_stem_midis):
    out_file = tmp_path / "all_stems.mid"
    res = build_combined_midi_project(sample_stem_midis, out_file, bpm=128.0)
    assert Path(res).exists()
    assert res == out_file

    pm = pretty_midi.PrettyMIDI(str(out_file))
    assert len(pm.instruments) == 4
    names = [inst.name.lower() for inst in pm.instruments]
    assert "vocals" in names
    assert "drums" in names
    assert "bass" in names
    assert "other" in names

    # Drum track verification
    drum_inst = [inst for inst in pm.instruments if inst.name.lower() == "drums"][0]
    assert drum_inst.is_drum is True

    # Bass track verification
    bass_inst = [inst for inst in pm.instruments if inst.name.lower() == "bass"][0]
    assert bass_inst.is_drum is False
    assert bass_inst.program in (32, 33, 34)  # GM Bass programs

def test_build_combined_midi_empty_stems(tmp_path: Path):
    out_file = tmp_path / "empty_project.mid"
    res = build_combined_midi_project({}, out_file, bpm=120.0)
    assert res.exists()
    pm = pretty_midi.PrettyMIDI(str(res))
    assert len(pm.instruments) == 0
