from pathlib import Path
import pretty_midi
import pytest
from music_re.core.midi_tonal import clean_and_quantize_midi

def test_clean_and_quantize_midi(tmp_path: Path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    # Add short glitch (10 ms)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.01))
    # Add valid note (start=0.51 -> snapped to 0.5 with 16th grid at 120 BPM)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.51, end=1.0))
    pm.instruments.append(inst)

    raw_mid = tmp_path / "raw.mid"
    pm.write(str(raw_mid))

    clean_mid = tmp_path / "clean.mid"
    clean_and_quantize_midi(raw_mid, clean_mid, bpm=120.0, min_duration=0.05)

    cleaned_pm = pretty_midi.PrettyMIDI(str(clean_mid))
    assert len(cleaned_pm.instruments[0].notes) == 1
    assert cleaned_pm.instruments[0].notes[0].pitch == 64
    # Check snapping: 0.51 -> 0.5s at 120 BPM (16th note = 0.125s)
    assert abs(cleaned_pm.instruments[0].notes[0].start - 0.5) < 0.02
