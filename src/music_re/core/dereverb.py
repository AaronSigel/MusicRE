import numpy as np
import librosa


def suppress_reverb_and_tails(
    y: np.ndarray,
    sr: int = 22050,
    decay_threshold_db: float = -24.0,
    release_ms: float = 80.0,
) -> np.ndarray:
    """Lightweight acoustic transient gating and spectral tail attenuation.

    Attenuates lingering reverberant tails, delay trails, and diffuse noise
    floors that fall below `decay_threshold_db` relative to the local peak
    following a transient onset, while preserving clear attack peaks and
    prominent harmonic content.

    Parameters
    ----------
    y : np.ndarray
        Audio signal array. Supports mono `(n,)` and stereo `(n, 2)` or `(2, n)`.
    sr : int, default=22050
        Sampling rate in Hz.
    decay_threshold_db : float, default=-24.0
        Threshold in dB relative to local peak below which decaying tails are attenuated.
    release_ms : float, default=80.0
        Release smoothing time in milliseconds for gate decay transitions.

    Returns
    -------
    np.ndarray
        Processed audio array with reverb tails suppressed, matching input shape.
    """
    if y.size == 0:
        return y.copy()

    if not np.all(np.isfinite(y)):
        raise ValueError("Audio data contains NaN or Inf values")

    orig_shape = y.shape
    if y.ndim == 1:
        channels = [y]
    elif y.ndim == 2:
        if y.shape[1] == 2:
            channels = [y[:, 0], y[:, 1]]
        elif y.shape[0] == 2:
            channels = [y[0, :], y[1, :]]
        else:
            channels = [y[:, c] for c in range(y.shape[1])]
    else:
        raise ValueError(f"Unsupported audio array shape: {y.shape}")

    processed = []
    for ch in channels:
        if not np.any(ch):
            processed.append(ch.copy().astype(np.float32))
            continue

        n_samples = len(ch)
        if n_samples < 32:
            processed.append(ch.copy().astype(np.float32))
            continue

        # Short-block envelope tracking (~5ms hop, ~10ms block)
        hop_length = max(32, int(sr * 0.005))
        block_size = max(64, hop_length * 2)
        dt = hop_length / sr

        n_blocks = int(np.ceil(n_samples / hop_length))
        pad_len = max(0, n_blocks * hop_length + block_size - n_samples)
        padded = np.pad(ch, (0, pad_len), mode="constant")

        # 1. Compute block-level peak energy envelope
        blocks = np.lib.stride_tricks.sliding_window_view(
            padded[: n_blocks * hop_length + block_size], block_size
        )[::hop_length]
        block_peaks = np.max(np.abs(blocks), axis=-1)

        # 2. Local peak tracker (tracks recent attack transients with 1.2s memory)
        tau_peak = 1.2
        alpha_peak = np.exp(-dt / tau_peak)
        local_peak = np.zeros_like(block_peaks)
        p = 0.0
        for i in range(len(block_peaks)):
            p = max(block_peaks[i], p * alpha_peak)
            local_peak[i] = p

        # Local decay threshold in linear amplitude
        thresh_linear = 10.0 ** (decay_threshold_db / 20.0)
        thresh = local_peak * thresh_linear
        eps = 1e-9

        # 3. Dynamic gate gain based on envelope
        ratio = np.maximum(block_peaks, eps) / np.maximum(thresh, eps)
        # Downward expansion for lingering tails below threshold
        exp_slope = 3.0
        gain_raw = np.where(ratio >= 1.0, 1.0, np.clip(ratio**exp_slope, 0.001, 1.0))

        # Gate release smoothing (instant attack, release_ms decay)
        alpha_rel = np.exp(-dt / max(release_ms / 1000.0, 1e-4))
        gain_smooth = np.zeros_like(gain_raw)
        curr_g = 1.0
        for i in range(len(gain_raw)):
            g = gain_raw[i]
            if g >= curr_g:
                curr_g = g
            else:
                curr_g = curr_g * alpha_rel + (1.0 - alpha_rel) * g
            gain_smooth[i] = curr_g

        # 4. Spectral tail attenuation via STFT
        n_fft = min(1024, max(block_size * 2, 128))
        n_fft = 1 << (n_fft - 1).bit_length()
        win_length = n_fft

        D = librosa.stft(
            ch.astype(np.float32),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        mag = np.abs(D)
        phase = np.angle(D)

        # Diffuse noise floor per frame (median spectral energy across bins)
        spec_median = np.median(mag, axis=0, keepdims=True)
        harmonic_ratio = np.maximum(mag, eps) / np.maximum(spec_median, eps)
        harmonic_mask = np.clip((harmonic_ratio - 1.2) / 1.5, 0.0, 1.0)

        # Align gain frames with STFT frames
        n_frames = mag.shape[1]
        n_gain = len(gain_smooth)
        if n_frames == n_gain:
            g_frames = gain_smooth
        else:
            g_frames = np.interp(
                np.linspace(0.0, 1.0, n_frames),
                np.linspace(0.0, 1.0, n_gain),
                gain_smooth,
            )
        g_broad = g_frames[np.newaxis, :]

        # In tail regions where broadband gain drops, diffuse reverb floor is attenuated,
        # but strong harmonic peaks are preserved
        gain_spec = np.maximum(g_broad, harmonic_mask * np.maximum(g_broad, 0.5))
        gain_final = np.clip(gain_spec * (0.2 + 0.8 * g_broad), 0.001, 1.0)

        D_gated = (mag * gain_final) * np.exp(1j * phase)
        out_ch = librosa.istft(
            D_gated,
            hop_length=hop_length,
            win_length=win_length,
            length=n_samples,
        )

        # 5. Attack transient peak preservation in time domain
        sample_thresh = np.interp(
            np.arange(n_samples),
            np.arange(len(thresh)) * hop_length,
            thresh,
        )
        attack_mask = np.abs(ch) >= sample_thresh
        out_ch[attack_mask] = ch[attack_mask]

        processed.append(out_ch.astype(np.float32))

    if orig_shape == (len(processed[0]),):
        return processed[0]
    elif len(orig_shape) == 2 and orig_shape[1] == 2:
        return np.column_stack(processed)
    elif len(orig_shape) == 2 and orig_shape[0] == 2:
        return np.row_stack(processed)
    else:
        return np.column_stack(processed)
