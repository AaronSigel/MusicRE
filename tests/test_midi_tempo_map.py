import pytest
import pretty_midi
import numpy as np
from pathlib import Path
from music_re.schemas import MusicalTimeline
from music_re.core.midi_project import build_combined_midi_project
from music_re.core.midi_quantize import quantize_to_timeline

def test_tempo_map_accuracy_under_variable_tempo(tmp_path):
    # Timeline with 120 -> 60 BPM transition
    # beat 0 at 0.0s, beat 1 at 0.5s (120 bpm), beat 2 at 1.0s (120 bpm),
    # beat 3 at 2.0s (60 bpm), beat 4 at 3.0s (60 bpm)
    beats = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    downbeats = [0.0, 2.0]
    timeline = MusicalTimeline(
        duration_sec=5.0,
        bpm=120.0,
        beats=beats,
        downbeats=downbeats,
        sections=[]
    )
    
    # Create input midi for bass
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=33, is_drum=False, name="Bass")
    # Note at 2.0s (which is beat 3)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=40, start=2.0, end=2.5))
    pm.instruments.append(inst)
    
    bass_mid = tmp_path / "bass.mid"
    pm.write(str(bass_mid))
    
    combined_path = tmp_path / "combined.mid"
    build_combined_midi_project({"bass": bass_mid}, combined_path, timeline=timeline)
    
    # Read back and verify timing
    read_pm = pretty_midi.PrettyMIDI(str(combined_path))
    assert len(read_pm.instruments[0].notes) == 1
    n = read_pm.instruments[0].notes[0]
    assert abs(n.start - 2.0) < 0.005 # within 5 ms

def test_missing_stem_midi_raises_value_error(tmp_path):
    missing_mid = tmp_path / "nonexistent.mid"
    combined_path = tmp_path / "combined.mid"
    with pytest.raises(ValueError, match="nonexistent.mid|Missing"):
        build_combined_midi_project({"drums": missing_mid}, combined_path)

def test_quantize_grid_subdivisions_cap():
    from music_re.core.midi_quantize import quantize_to_timeline
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.1, end=0.4))
    pm.instruments.append(inst)
    timeline = MusicalTimeline(duration_sec=2.0, bpm=120.0, beats=[0.0, 0.5, 1.0], downbeats=[0.0], sections=[])
    with pytest.raises(ValueError, match="grid_subdivisions"):
        quantize_to_timeline(pm, timeline, grid_subdivisions=10000)
