from pathlib import Path
import pretty_midi
import pytest

from music_re.core.midi_tonal import clean_and_quantize_midi, transcribe_tonal

def test_clean_and_quantize_midi_polyphonic_overlap(tmp_path: Path):
    in_midi = tmp_path / "in.mid"
    out_midi = tmp_path / "out.mid"

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    # Note 1: Pitch 60, 0.0 to 2.0s
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=2.0))
    # Note 2: Pitch 64, 0.5 to 1.5s (intervening note)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=64, start=0.5, end=1.5))
    # Note 3: Pitch 60, 1.0 to 2.5s (overlaps with Note 1 of same pitch!)
    inst.notes.append(pretty_midi.Note(velocity=110, pitch=60, start=1.0, end=2.5))
    pm.instruments.append(inst)
    pm.write(str(in_midi))

    clean_and_quantize_midi(in_midi, out_midi, bpm=120.0)

    res_pm = pretty_midi.PrettyMIDI(str(out_midi))
    notes_p60 = [n for n in res_pm.instruments[0].notes if n.pitch == 60]

    # Verify no overlapping intervals for pitch 60
    for i in range(len(notes_p60) - 1):
        assert notes_p60[i].end <= notes_p60[i + 1].start, (
            f"Note {i} end ({notes_p60[i].end}) exceeds note {i+1} start ({notes_p60[i+1].start})"
        )

def test_transcribe_tonal_saves_raw_copy(tmp_path: Path):
    # Create simple 1s tone
    import numpy as np
    import soundfile as sf
    tone_wav = tmp_path / "tone.wav"
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    sf.write(str(tone_wav), y, sr)

    out_midi = tmp_path / "vocals.mid"
    raw_midi = tmp_path / "vocals_raw.mid"

    res = transcribe_tonal(tone_wav, out_midi, bpm=120.0)
    assert Path(res).exists()
    assert raw_midi.exists(), "Raw unquantized MIDI must be preserved"
