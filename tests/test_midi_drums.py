from pathlib import Path
import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from music_re.core.midi_drums import transcribe_drums

def test_transcribe_drums_basic(tmp_path: Path):
    sr = 22050
    duration = 2.0
    audio = np.zeros(int(sr * duration))

    # Low frequency pulse (kick ~60Hz) at 0.0s
    t_pulse = np.linspace(0, 0.08, int(sr * 0.08), endpoint=False)
    kick_pulse = 0.8 * np.sin(2 * np.pi * 60 * t_pulse)
    audio[0:len(t_pulse)] += kick_pulse

    # High frequency pulse (hi-hat ~8000Hz) at 0.5s
    hihat_pulse = 0.8 * np.sin(2 * np.pi * 8000 * t_pulse)
    idx_500 = int(0.5 * sr)
    audio[idx_500:idx_500 + len(t_pulse)] += hihat_pulse

    wav = tmp_path / "drums.wav"
    sf.write(str(wav), audio, sr)

    mid = tmp_path / "drums.mid"
    res = transcribe_drums(wav, mid, bpm=120.0)
    assert res == mid
    assert mid.exists()

    pm = pretty_midi.PrettyMIDI(str(mid))
    assert len(pm.instruments) > 0
    drum_inst = pm.instruments[0]
    assert drum_inst.is_drum
    assert len(drum_inst.notes) >= 2

    # Check pitches contain kick (36) and hihat (42)
    pitches = [n.pitch for n in drum_inst.notes]
    assert 36 in pitches
    assert 42 in pitches

def test_transcribe_drums_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        transcribe_drums(tmp_path / "not_there.wav", tmp_path / "out.mid")
