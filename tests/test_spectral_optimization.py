import numpy as np
import librosa
import warnings
from dataclasses import replace
import time
import tracemalloc

from music_re.core.chords import CHORD_TEMPLATES, detect_chords_for_bars
from music_re.core import spectral
from music_re.core.midi_validation import compute_note_evidence


def synthetic_stem(seconds, sr=22050):
    t = np.arange(sr * seconds, dtype=np.float32) / sr
    return (0.3 * np.sin(2 * np.pi * (60 * t + 0.4 * t * t))
            + 0.2 * np.sin(2 * np.pi * 220 * t)
            + 0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_chunked_features_and_note_evidence_match_whole():
    config = spectral.SpectralConfig(chunk_sec=60)
    y = synthetic_stem(150)
    chunked = spectral.compute_stem_features(y, config)
    whole = spectral.compute_stem_features(y, replace(config, chunk_sec=0))
    for name in ('times', 'stft_freqs', 'cqt_midi'):
        np.testing.assert_allclose(getattr(chunked, name), getattr(whole, name),
                                   rtol=0, atol=0, err_msg=name)
    for name in ('stft_mag', 'cqt_mag', 'rms', 'flux'):
        expected = getattr(whole, name)
        assert getattr(chunked, name).dtype == np.float32
        np.testing.assert_allclose(getattr(chunked, name), expected, rtol=1e-3,
                                   atol=1e-4 * np.abs(expected).max(), err_msg=name)
    boundary = (int(config.chunk_sec * config.sr / config.hop_length) // 64
                * 64 * config.hop_length / config.sr)
    for start in (1, boundary - 0.2, boundary + 0.1,
                  2 * boundary - 0.2, 2 * boundary + 0.1, 148):
        a = compute_note_evidence(chunked, 57, start, start + 0.4)
        b = compute_note_evidence(whole, 57, start, start + 0.4)
        for field in ('harmonic_support', 'spectral_contrast', 'temporal_persistence',
                      'onset_agreement', 'leakage'):
            np.testing.assert_allclose(getattr(a, field), getattr(b, field), rtol=0, atol=1e-3)


def test_long_stem_scaling_and_memory():
    config = spectral.SpectralConfig()
    spectral.compute_stem_features(synthetic_stem(1), config)
    times = []
    for seconds in (5, 30, 180):
        y = synthetic_stem(seconds)
        tracemalloc.start()
        start = time.perf_counter()
        features = spectral.compute_stem_features(y, config)
        times.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if seconds == 180:
            assert peak < 520 * 1024 * 1024
        del features, y
    assert times[2] / times[1] < 9


def test_long_stem_uses_bounded_chunks(monkeypatch):
    config = spectral.SpectralConfig(chunk_sec=2)
    y = np.zeros(config.sr * 12, dtype=np.float32)
    original = spectral.compute_cqt
    lengths = []

    def counted(signal, *args, **kwargs):
        lengths.append(len(signal))
        return original(signal, *args, **kwargs)

    monkeypatch.setattr(spectral, 'compute_cqt', counted)
    result = spectral.compute_stem_features(y, config)
    assert len(lengths) > 1
    assert max(lengths) < len(y)
    assert len(result.times) == 1 + len(y) // config.hop_length


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
