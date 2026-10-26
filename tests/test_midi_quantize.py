import pretty_midi
import pytest

from music_re.core.midi_quantize import quantize_to_timeline
from music_re.schemas import MusicalTimeline

def test_quantize_static_bpm_fallback():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    # At 120 bpm, 1 beat = 0.5s, 16th note = 0.125s
    # Note starts at 0.11s -> should snap to 0.125s
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.11, end=0.4))
    pm.instruments.append(inst)

    quantized = quantize_to_timeline(pm, timeline=None, bpm=120.0, grid_subdivisions=4)
    assert len(quantized.instruments[0].notes) == 1
    note = quantized.instruments[0].notes[0]
    assert note.start == pytest.approx(0.125, abs=1e-3)
    assert note.end == pytest.approx(0.375, abs=1e-3)
    assert note.end >= note.start + 0.05

def test_quantize_dynamic_timeline_drifty_beats():
    # Beats with tempo drift: beat 0 at 0.0s, beat 1 at 0.8s
    # Subdivisions (4): 0.0, 0.2, 0.4, 0.6, 0.8
    tl = MusicalTimeline(
        duration_sec=2.0,
        bpm=75.0,
        beats=[0.0, 0.8, 1.6],
        downbeats=[0.0],
        sections=[],
    )

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    # Note onset at 0.19s -> snaps to dynamic 0.2s subdivision
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.19, end=0.7))
    pm.instruments.append(inst)

    quantized = quantize_to_timeline(pm, timeline=tl, grid_subdivisions=4)
    note = quantized.instruments[0].notes[0]
    assert note.start == pytest.approx(0.2, abs=1e-3)

def test_quantize_eliminates_polyphonic_overlaps():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    # Pitch 60: 0.0 -> 1.0s
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    # Intervening pitch 65: 0.2 -> 0.8s
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=65, start=0.2, end=0.8))
    # Pitch 60 overlaps: 0.5 -> 1.5s
    inst.notes.append(pretty_midi.Note(velocity=110, pitch=60, start=0.5, end=1.5))
    pm.instruments.append(inst)

    quantized = quantize_to_timeline(pm, timeline=None, bpm=120.0)
    p60_notes = [n for n in quantized.instruments[0].notes if n.pitch == 60]
    for i in range(len(p60_notes) - 1):
        assert p60_notes[i].end <= p60_notes[i + 1].start

def test_quantize_caps_at_timeline_duration():
    tl = MusicalTimeline(
        duration_sec=2.0,
        bpm=120.0,
        beats=[0.0, 0.5, 1.0, 1.5],
        downbeats=[0.0],
        sections=[],
    )
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    # Note starts right before track end: 1.96s -> 2.05s
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=1.96, end=2.05))
    # Note starts after track end
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=62, start=2.1, end=2.3))
    pm.instruments.append(inst)

    quantized = quantize_to_timeline(pm, timeline=tl, grid_subdivisions=4)
    notes = quantized.instruments[0].notes
    assert len(notes) == 1
    assert notes[0].end <= 2.0 + 1e-4
    assert notes[0].start < 2.0


def test_pickup_and_duplicate_beats():
    from music_re.core.midi_quantize import build_quantize_grid
    tl = MusicalTimeline(duration_sec=3, bpm=120, beats=[1, 1, 1.5, 2], downbeats=[])
    grid = build_quantize_grid(tl, grid_subdivisions=4, max_duration=3)
    assert 0.0 in grid and 0.125 in grid
    assert all(b > a for a, b in zip(grid, grid[1:]))


@pytest.mark.parametrize('subdivisions', [0, -1, 1.5])
def test_invalid_grid_rejected(subdivisions):
    from music_re.core.midi_quantize import build_quantize_grid
    with pytest.raises(ValueError, match='subdivisions'):
        build_quantize_grid(grid_subdivisions=subdivisions)


def test_onset_only_keeps_original_end():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(100, 60, .11, .4)]
    pm.instruments.append(inst)
    result = quantize_to_timeline(pm, quantize_ends=False)
    assert result.instruments[0].notes[0].end == .4
