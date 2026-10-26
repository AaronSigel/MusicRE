from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from music_re.core.backends.structure_base import StructureAnalyzerBackend
from music_re.core.backends.librosa_structure_backend import LibrosaStructureAnalyzer
from music_re.schemas import MusicalTimeline

def test_structure_base_cannot_instantiate():
    with pytest.raises(TypeError):
        StructureAnalyzerBackend("base")

def test_librosa_structure_analyzer(tmp_path: Path):
    sr = 22050
    duration_sec = 8.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # 2 Hz beat pulses = 120 bpm
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t) * (np.sin(2 * np.pi * 2.0 * t) > 0.9)
    track_wav = tmp_path / "structure_test.wav"
    sf.write(str(track_wav), y, sr)

    analyzer = LibrosaStructureAnalyzer()
    timeline = analyzer.analyze(track_wav)

    assert isinstance(timeline, MusicalTimeline)
    assert timeline.duration_sec == pytest.approx(duration_sec, abs=0.1)
    assert timeline.bpm > 0
    assert len(timeline.beats) > 0
    assert timeline.downbeats == []
    assert all(s.label == 'unknown' and s.confidence is None for s in timeline.sections)
    assert len(timeline.sections) > 0

    # Verify sections are contiguous and non-overlapping
    assert timeline.sections[0].start_sec == 0.0
    for i in range(len(timeline.sections) - 1):
        assert timeline.sections[i].end_sec == timeline.sections[i + 1].start_sec
    assert timeline.sections[-1].end_sec == pytest.approx(timeline.duration_sec, abs=0.1)


def test_segmentation_failure_does_not_invent_boundaries(tmp_path, monkeypatch):
    import librosa
    path = tmp_path / 'long.wav'
    sf.write(path, np.random.default_rng(1).normal(0, .1, 22050 * 15), 22050)
    def fail(*args, **kwargs):
        raise ValueError('segmentation failed')
    monkeypatch.setattr(librosa.segment, 'agglomerative', fail)
    timeline = LibrosaStructureAnalyzer().analyze(path)
    assert len(timeline.sections) == 1
    assert timeline.sections[0].label == 'unknown'
    assert timeline.sections[0].confidence is None
    assert timeline.sections[0].end_sec == 15
    assert timeline.warnings
