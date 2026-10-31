import numpy as np
import librosa
import warnings

from music_re.core.chords import CHORD_TEMPLATES, detect_chords_for_bars


def old_bar_chord(y, sr):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean = chroma.mean(axis=1)
    mean /= np.linalg.norm(mean)
    label, score = max(((label, float(np.dot(mean, template)))
                        for label, template in CHORD_TEMPLATES.items()),
                       key=lambda item: item[1])
    return label, round(max(0, score), 2)


def test_chroma_is_computed_once_for_progression(monkeypatch):
    sr = 22050
    t = np.arange(2 * sr) / sr
    notes = [(60, 64, 67), (67, 71, 74), (69, 72, 76), (65, 69, 72)]
    y = np.concatenate([
        sum(np.sin(2 * np.pi * librosa.midi_to_hz(n) * t) for n in triad)
        for triad in notes
    ]).astype(np.float32)
    bars = [(2 * i, 2 * (i + 1)) for i in range(4)]
    old = [old_bar_chord(y[a * sr:b * sr], sr) for a, b in bars]
    original = librosa.feature.chroma_cqt
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(librosa.feature, 'chroma_cqt', counted)
    result = detect_chords_for_bars(y, sr, bars)
    assert len(calls) == 1
    assert [x['chord'] for x in result] == [x[0] for x in old]
    np.testing.assert_allclose([x['confidence'] for x in result],
                               [x[1] for x in old], atol=0.05)
