from pathlib import Path
import pretty_midi
import pytest
from music_re.core.midi_digest import generate_midi_digest
from music_re.schemas import MidiDigest

def test_generate_midi_digest(tmp_path: Path):
    bpm = 120.0  # 0.5s per beat, 2.0s per 4/4 bar
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0)

    # Bar 1 Beat 1 (t=0.0s) -> bar 1, beat 1.0, duration 1 beat (0.5s)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5))

    # Bar 2 Beat 3 (t=3.0s: 6 beats total -> bar 2, beat 3.0)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=62, start=3.0, end=3.5))

    # Bar 10 Note (t=18.0s) -> should be filtered out if bars 1-8 requested
    inst.notes.append(pretty_midi.Note(velocity=70, pitch=64, start=18.0, end=18.5))

    pm.instruments.append(inst)
    mid_file = tmp_path / "test.mid"
    pm.write(str(mid_file))

    digest = generate_midi_digest(mid_file, bpm=bpm, start_bar=1, end_bar=2)
    assert isinstance(digest, MidiDigest)
    assert digest.total_notes == 2
    assert digest.bars_analyzed == "1-2"
    assert digest.bpm == 120.0

    n1 = digest.notes[0]
    assert n1.bar == 1
    assert n1.beat_in_bar == 1.0
    assert n1.note == "C4"
    assert n1.pitch == 60
    assert n1.velocity == 90
    assert n1.duration_beats == 1.0

    n2 = digest.notes[1]
    assert n2.bar == 2
    assert n2.beat_in_bar == 3.0
    assert n2.note == "D4"
    assert n2.pitch == 62

def test_generate_midi_digest_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        generate_midi_digest(tmp_path / "nonexistent.mid", bpm=120.0)
