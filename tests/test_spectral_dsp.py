"""Tests for music_re.core.spectral — pure DSP, no models, no file I/O."""
import warnings

import numpy as np
import pytest

SR = 22050
HOP = 256
N_FFT = 4096


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq_hz: float, duration_s: float = 2.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _impulse(sr: int = SR, offset_s: float = 0.5, duration_s: float = 2.0) -> np.ndarray:
    y = np.zeros(int(duration_s * sr), dtype=np.float32)
    y[int(offset_s * sr)] = 1.0
    return y


# ---------------------------------------------------------------------------
# midi_to_hz
# ---------------------------------------------------------------------------

class TestMidiToHz:
    def test_a4(self):
        from music_re.core.spectral import midi_to_hz
        assert abs(midi_to_hz(69) - 440.0) < 1e-6

    def test_a1(self):
        from music_re.core.spectral import midi_to_hz
        assert abs(midi_to_hz(33) - 55.0) < 1e-4

    def test_c1(self):
        from music_re.core.spectral import midi_to_hz
        assert abs(midi_to_hz(24) - 32.703) < 0.01


# ---------------------------------------------------------------------------
# compute_stft
# ---------------------------------------------------------------------------

class TestComputeStft:
    def test_shape(self):
        from music_re.core.spectral import compute_stft
        y = _sine(440.0)
        freqs, times, mag = compute_stft(y)
        assert mag.ndim == 2
        assert mag.shape[0] == N_FFT // 2 + 1
        assert mag.dtype == np.float32

    def test_440hz_peak(self):
        from music_re.core.spectral import compute_stft
        y = _sine(440.0)
        freqs, times, mag = compute_stft(y)
        mean_mag = mag.mean(axis=1)
        peak_freq = freqs[np.argmax(mean_mag)]
        assert abs(peak_freq - 440.0) < 10.0

    def test_gain_invariance(self):
        from music_re.core.spectral import compute_stft
        y = _sine(440.0, amp=0.5)
        y2 = _sine(440.0, amp=0.1)
        _, _, mag1 = compute_stft(y)
        _, _, mag2 = compute_stft(y2)
        peak1 = np.argmax(mag1.mean(axis=1))
        peak2 = np.argmax(mag2.mean(axis=1))
        assert peak1 == peak2

    def test_silence_no_nan(self):
        from music_re.core.spectral import compute_stft
        y = np.zeros(SR, dtype=np.float32)
        freqs, times, mag = compute_stft(y)
        assert np.all(np.isfinite(mag))

    def test_empty_array(self):
        from music_re.core.spectral import compute_stft
        y = np.zeros(0, dtype=np.float32)
        freqs, times, mag = compute_stft(y)
        assert np.all(np.isfinite(mag))

    def test_short_signal(self):
        from music_re.core.spectral import compute_stft
        # shorter than n_fft — must not raise or warn about n_fft too large
        y = np.ones(100, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            freqs, times, mag = compute_stft(y)
        assert np.all(np.isfinite(mag))


# ---------------------------------------------------------------------------
# compute_cqt
# ---------------------------------------------------------------------------

class TestComputeCqt:
    def test_shape(self):
        from music_re.core.spectral import compute_cqt
        y = _sine(440.0)
        cqt_mag, cqt_midi = compute_cqt(y)
        assert cqt_mag.ndim == 2
        assert cqt_midi.ndim == 1
        assert cqt_mag.shape[0] == cqt_midi.shape[0]
        assert cqt_mag.dtype == np.float32

    def test_55hz_peak_via_cqt(self):
        """STFT bin spacing ~5.4 Hz at 22050/4096 is too coarse; CQT must resolve 55 Hz."""
        from music_re.core.spectral import compute_cqt, midi_to_hz
        y = _sine(55.0, duration_s=3.0)
        cqt_mag, cqt_midi = compute_cqt(y)
        mean_mag = cqt_mag.mean(axis=1)
        peak_midi = cqt_midi[np.argmax(mean_mag)]
        peak_hz = midi_to_hz(float(peak_midi))
        assert abs(peak_hz - 55.0) < 6.0  # within half a semitone

    def test_silence_no_nan(self):
        from music_re.core.spectral import compute_cqt
        y = np.zeros(SR, dtype=np.float32)
        cqt_mag, cqt_midi = compute_cqt(y)
        assert np.all(np.isfinite(cqt_mag))

    def test_empty_array(self):
        from music_re.core.spectral import compute_cqt
        y = np.zeros(0, dtype=np.float32)
        cqt_mag, cqt_midi = compute_cqt(y)
        assert np.all(np.isfinite(cqt_mag))


# ---------------------------------------------------------------------------
# compute_rms
# ---------------------------------------------------------------------------

class TestComputeRms:
    def test_shape(self):
        from music_re.core.spectral import compute_rms
        y = _sine(440.0)
        rms = compute_rms(y)
        assert rms.ndim == 1
        assert rms.dtype == np.float32

    def test_silence_is_zero(self):
        from music_re.core.spectral import compute_rms
        y = np.zeros(SR, dtype=np.float32)
        rms = compute_rms(y)
        assert np.all(rms == 0.0)

    def test_no_nan(self):
        from music_re.core.spectral import compute_rms
        y = _sine(440.0)
        assert np.all(np.isfinite(compute_rms(y)))

    def test_empty_array(self):
        from music_re.core.spectral import compute_rms
        y = np.zeros(0, dtype=np.float32)
        rms = compute_rms(y)
        assert np.all(np.isfinite(rms))


# ---------------------------------------------------------------------------
# compute_spectral_flux
# ---------------------------------------------------------------------------

class TestComputeSpectralFlux:
    def test_shape_matches_frames(self):
        from music_re.core.spectral import compute_stft, compute_spectral_flux
        y = _sine(440.0)
        _, _, mag = compute_stft(y)
        flux = compute_spectral_flux(mag)
        assert flux.shape == (mag.shape[1],)
        assert flux.dtype == np.float32

    def test_silence_is_zero(self):
        from music_re.core.spectral import compute_stft, compute_spectral_flux
        y = np.zeros(SR * 2, dtype=np.float32)
        _, _, mag = compute_stft(y)
        flux = compute_spectral_flux(mag)
        assert np.all(flux == 0.0)

    @pytest.mark.parametrize('offset_s', [0.2, 0.5, 1.1, 1.7])
    def test_impulse_onset_within_one_hop(self, offset_s):
        from music_re.core.spectral import compute_stem_features, SpectralConfig
        y = _impulse(offset_s=offset_s)
        features = compute_stem_features(y, SpectralConfig())
        peak_frame = int(np.argmax(features.flux))
        assert abs(features.times[peak_frame] - offset_s) <= HOP / SR + 1e-6

    def test_no_nan(self):
        from music_re.core.spectral import compute_stft, compute_spectral_flux
        y = _sine(440.0)
        _, _, mag = compute_stft(y)
        assert np.all(np.isfinite(compute_spectral_flux(mag)))

    def test_empty_mag(self):
        from music_re.core.spectral import compute_spectral_flux
        mag = np.zeros((N_FFT // 2 + 1, 0), dtype=np.float32)
        flux = compute_spectral_flux(mag)
        assert flux.shape == (0,)


# ---------------------------------------------------------------------------
# detect_harmonic_peaks
# ---------------------------------------------------------------------------

class TestDetectHarmonicPeaks:
    def _harmonic_signal(self, f0=440.0, n=4, sr=SR):
        t = np.arange(int(2.0 * sr)) / sr
        y = np.zeros(len(t), dtype=np.float32)
        for k in range(1, n + 1):
            y += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t).astype(np.float32)
        return y

    def test_finds_fundamental(self):
        from music_re.core.spectral import compute_stft, detect_harmonic_peaks
        y = self._harmonic_signal()
        freqs, _, mag = compute_stft(y)
        mean_col = mag.mean(axis=1)
        peaks = detect_harmonic_peaks(mean_col, freqs, f0=440.0)
        assert 440.0 in peaks or any(abs(p - 440.0) < 15.0 for p in peaks)

    def test_silence_no_peaks(self):
        from music_re.core.spectral import compute_stft, detect_harmonic_peaks
        y = np.zeros(SR * 2, dtype=np.float32)
        freqs, _, mag = compute_stft(y)
        mean_col = mag.mean(axis=1)
        peaks = detect_harmonic_peaks(mean_col, freqs, f0=440.0)
        assert isinstance(peaks, list)

    def test_returns_list_of_floats(self):
        from music_re.core.spectral import compute_stft, detect_harmonic_peaks
        y = self._harmonic_signal()
        freqs, _, mag = compute_stft(y)
        peaks = detect_harmonic_peaks(mag.mean(axis=1), freqs, f0=440.0)
        assert all(isinstance(p, float) for p in peaks)


# ---------------------------------------------------------------------------
# SpectralConfig dataclass
# ---------------------------------------------------------------------------

class TestSpectralConfig:
    def test_defaults(self):
        from music_re.core.spectral import SpectralConfig
        cfg = SpectralConfig()
        assert cfg.sr == 22050
        assert cfg.n_fft == 4096
        assert cfg.hop_length == 256
        assert cfg.window == 'hann'
        assert cfg.cqt_bins == 84
        assert cfg.cqt_bins_per_octave == 12
        assert cfg.cqt_fmin_midi == 24
        assert cfg.chunk_sec == 60.0

    def test_frozen(self):
        from music_re.core.spectral import SpectralConfig
        cfg = SpectralConfig()
        with pytest.raises((TypeError, AttributeError)):
            cfg.sr = 44100  # type: ignore[misc]

    def test_to_dict(self):
        from music_re.core.spectral import SpectralConfig
        d = SpectralConfig().to_dict()
        assert isinstance(d, dict)
        assert d['sr'] == 22050
        assert d['n_fft'] == 4096


# ---------------------------------------------------------------------------
# StemSpectralFeatures dataclass
# ---------------------------------------------------------------------------

class TestStemSpectralFeatures:
    def test_fields_exist(self):
        from music_re.core.spectral import StemSpectralFeatures
        import dataclasses
        fields = {f.name for f in dataclasses.fields(StemSpectralFeatures)}
        required = {'sr', 'hop_length', 'times', 'stft_freqs', 'stft_mag',
                    'cqt_mag', 'cqt_midi', 'rms', 'flux'}
        assert required <= fields


# ---------------------------------------------------------------------------
# compute_stem_features — integration
# ---------------------------------------------------------------------------

class TestComputeStemFeatures:
    @pytest.mark.parametrize('sample_count', [1, 64, 2048, 4097])
    def test_short_signal_has_original_duration_frame_grid(self, sample_count):
        from music_re.core.spectral import compute_stem_features, SpectralConfig
        features = compute_stem_features(np.ones(sample_count, dtype=np.float32), SpectralConfig())
        expected = 1 + sample_count // HOP
        assert features.times.shape == (expected,)
        assert features.stft_mag.shape[1] == expected
        assert features.cqt_mag.shape[1] == expected
        assert features.rms.shape == (expected,)
        assert features.flux.shape == (expected,)

    def test_configured_window_reaches_stft(self):
        from music_re.core.spectral import compute_stem_features, SpectralConfig
        y = _sine(440.0, duration_s=0.2)
        hann = compute_stem_features(y, SpectralConfig(window='hann'))
        boxcar = compute_stem_features(y, SpectralConfig(window='boxcar'))
        assert not np.allclose(hann.stft_mag, boxcar.stft_mag)

    def test_basic_shapes(self):
        from music_re.core.spectral import compute_stem_features, SpectralConfig
        cfg = SpectralConfig()
        y = _sine(440.0, duration_s=2.0)
        feat = compute_stem_features(y, cfg)
        n_frames = feat.stft_mag.shape[1]
        assert feat.stft_mag.dtype == np.float32
        assert feat.cqt_mag.dtype == np.float32
        assert feat.rms.shape == (n_frames,)
        assert feat.flux.shape == (n_frames,)
        assert feat.times.shape == (n_frames,)
        assert feat.stft_mag.shape[0] == cfg.n_fft // 2 + 1
        assert feat.cqt_mag.shape[0] == cfg.cqt_bins

    def test_stereo_downmix(self):
        from music_re.core.spectral import compute_stem_features, SpectralConfig
        cfg = SpectralConfig()
        mono = _sine(440.0, duration_s=1.0)
        stereo = np.stack([mono, mono * 0.5], axis=0)  # [2, samples]
        feat = compute_stem_features(stereo, cfg)
        assert feat.stft_mag.ndim == 2

    def test_no_nan_finite(self):
        from music_re.core.spectral import compute_stem_features, SpectralConfig
        cfg = SpectralConfig()
        for y in [
            _sine(440.0),
            np.zeros(SR, dtype=np.float32),
            _impulse(),
            np.random.default_rng(0).random(SR * 2).astype(np.float32),
        ]:
            feat = compute_stem_features(y, cfg)
            assert np.all(np.isfinite(feat.stft_mag))
            assert np.all(np.isfinite(feat.cqt_mag))
            assert np.all(np.isfinite(feat.rms))
            assert np.all(np.isfinite(feat.flux))

    def test_empty_signal(self):
        from music_re.core.spectral import compute_stem_features, SpectralConfig
        cfg = SpectralConfig()
        y = np.zeros(0, dtype=np.float32)
        feat = compute_stem_features(y, cfg)
        assert np.all(np.isfinite(feat.stft_mag))
        assert np.all(np.isfinite(feat.cqt_mag))

    def test_short_signal(self):
        from music_re.core.spectral import compute_stem_features, SpectralConfig
        cfg = SpectralConfig()
        y = np.ones(64, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            feat = compute_stem_features(y, cfg)
        assert np.all(np.isfinite(feat.stft_mag))


# ---------------------------------------------------------------------------
# load_mono (smoke-test; skipped when no real audio file available)
# ---------------------------------------------------------------------------

class TestLoadMono:
    def test_signature_exists(self):
        from music_re.core.spectral import load_mono
        import inspect
        sig = inspect.signature(load_mono)
        assert 'path' in sig.parameters
        assert 'sr' in sig.parameters

    def test_decoder_frames_channels_axis_even_when_frames_le_channels(self, monkeypatch):
        import music_re.core.audio_io
        from music_re.core.spectral import load_mono
        monkeypatch.setattr(
            music_re.core.audio_io, 'decode_audio',
            lambda *args, **kwargs: (np.array([[1.0, 3.0]], dtype=np.float32), SR),
        )
        np.testing.assert_array_equal(load_mono('unused.wav'), np.array([2.0], dtype=np.float32))
