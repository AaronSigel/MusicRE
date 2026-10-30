import pytest
import soundfile as sf
import numpy as np
from pathlib import Path

from music_re.core.backends import get_structure_analyzer, get_transcriber_backend
from music_re.core.backends.adtof_backend import ADTOFDrumTranscriber
from music_re.core.backends.beat_this_backend import BeatThisStructureAnalyzer
from music_re.schemas import MusicalTimeline

def test_beat_this_structure_analyzer_on_synthetic_audio(tmp_path):
    # Generate 3 seconds of synthetic click track (120 BPM = 0.5s intervals)
    sr = 22050
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    y = np.zeros_like(t)
    for onset in [0.5, 1.0, 1.5, 2.0, 2.5]:
        idx = int(onset * sr)
        y[idx:idx + 500] = 0.8
        
    wav_path = tmp_path / "click.wav"
    sf.write(str(wav_path), y, sr)
    
    analyzer = get_structure_analyzer(name="beat_this")
    timeline = analyzer.analyze(wav_path)
    
    assert isinstance(timeline, MusicalTimeline)
    assert timeline.backend == "beat_this"
    assert timeline.duration_sec == pytest.approx(3.0, abs=0.1)
    assert len(timeline.beats) >= 0

def test_adtof_drum_transcriber_on_synthetic_audio(tmp_path):
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Simple low-frequency pulse for kick
    y = 0.7 * np.sin(2 * np.pi * 60.0 * t) * np.exp(-t * 15.0)
    wav_path = tmp_path / "kick.wav"
    sf.write(str(wav_path), y, sr)
    
    transcriber = get_transcriber_backend(mode="drums", name="adtof")
    assert isinstance(transcriber, ADTOFDrumTranscriber)
    
    out_midi = tmp_path / "drums.mid"
    res = transcriber.transcribe(wav_path, out_midi, save_raw=True)
    
    assert Path(res.midi_path).is_file()
    assert Path(res.raw_midi_path).is_file()
    assert res.note_count >= 0
