import gc
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from music_re.core.backends.separator_base import SeparatorBackend

SCNET_STEMS: List[str] = ["drums", "bass", "other", "vocals"]


class DualPathRNNBlock(nn.Module):
    """Dual-path RNN processing alternating across frequency (intra) and time (inter) axes."""

    def __init__(self, hidden_dim: int = 32, rnn_type: str = "LSTM"):
        super().__init__()
        rnn_cls = getattr(nn, rnn_type)
        self.intra_rnn = rnn_cls(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.intra_norm = nn.GroupNorm(1, hidden_dim)
        self.inter_rnn = rnn_cls(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.inter_norm = nn.GroupNorm(1, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        B, C, F_dim, T = x.shape
        # Intra-path: across frequency F
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * T, F_dim, C)
        intra_out, _ = self.intra_rnn(intra)
        intra_out = intra_out.view(B, T, F_dim, C).permute(0, 3, 2, 1).contiguous()
        x = x + self.intra_norm(intra_out)

        # Inter-path: across time T
        inter = x.permute(0, 2, 3, 1).contiguous().view(B * F_dim, T, C)
        inter_out, _ = self.inter_rnn(inter)
        inter_out = inter_out.view(B, F_dim, T, C).permute(0, 3, 1, 2).contiguous()
        x = x + self.inter_norm(inter_out)
        return x


class SparseDownsampler(nn.Module):
    """Sparse compression/downsampling along frequency bands."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(stride * 2, 1),
            stride=(stride, 1),
            padding=(stride // 2, 0),
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SparseUpsampler(nn.Module):
    """Sparse decompression/upsampling along frequency bands."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 4):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(stride * 2, 1),
            stride=(stride, 1),
            padding=(stride // 2, 0),
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor, target_f: int) -> torch.Tensor:
        out = self.act(self.norm(self.deconv(x)))
        if out.shape[2] != target_f:
            out = F.interpolate(out, size=(target_f, out.shape[3]), mode="bilinear", align_corners=False)
        return out


class SCNet(nn.Module):
    """Sparse Compression Network for 4-stem music separation.

    Operates in time-frequency domain with sparse frequency compression,
    dual-path RNN temporal/spectral processing, and sparse decompression.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_sources: int = 4,
        n_fft: int = 2048,
        hop_length: int = 512,
        hidden_dim: int = 32,
        num_dprnn_layers: int = 2,
        stride: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_sources = num_sources
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        feat_dim = in_channels * 2  # Real and imaginary parts
        self.downsampler = SparseDownsampler(feat_dim, hidden_dim, stride=stride)
        self.dprnn_blocks = nn.ModuleList([
            DualPathRNNBlock(hidden_dim) for _ in range(num_dprnn_layers)
        ])
        self.upsampler = SparseUpsampler(hidden_dim, hidden_dim, stride=stride)
        self.out_conv = nn.Conv2d(hidden_dim, num_sources * feat_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        window = self.window.to(x.device)

        x_reshaped = x.view(B * C, T)
        stft = torch.stft(
            x_reshaped,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        F_bins, T_frames = stft.shape[-2], stft.shape[-1]

        stft_real_imag = torch.stack([stft.real, stft.imag], dim=1)
        stft_feat = stft_real_imag.view(B, C * 2, F_bins, T_frames)

        compressed = self.downsampler(stft_feat)
        h = compressed
        for block in self.dprnn_blocks:
            h = block(h)
        decompressed = self.upsampler(h, target_f=F_bins)

        masks = self.out_conv(decompressed)
        masks = masks.view(B, self.num_sources, C, 2, F_bins, T_frames)

        stft_orig = stft.view(B, C, F_bins, T_frames)
        m_real = torch.tanh(masks[:, :, :, 0, :, :])
        m_imag = torch.tanh(masks[:, :, :, 1, :, :])
        m_complex = torch.complex(m_real, m_imag)

        out_stft = stft_orig.unsqueeze(1) * m_complex
        out_stft_flat = out_stft.view(B * self.num_sources * C, F_bins, T_frames)
        wavs = torch.istft(
            out_stft_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=T,
        )
        return wavs.view(B, self.num_sources, C, T)


class SCNetSeparator(SeparatorBackend):
    """Separator adapter using SCNet architecture."""

    def __init__(
        self,
        model_name: str = "scnet",
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(backend_name="scnet", model_name=model_name, device=device)
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.shifts = int(kwargs.get("shifts", 1))
        self.overlap = float(kwargs.get("overlap", 0.25))
        self._model: Optional[SCNet] = None
        self.quality_note = "SCNet 4-stem sparse compression separator (drums, bass, other, vocals)."

    @property
    def supported_stems(self) -> List[str]:
        return list(SCNET_STEMS)

    def load(self) -> None:
        """Lazy load SCNet model onto device."""
        if self._model is not None:
            return

        is_large = "large" in self.model_name.lower()
        hidden_dim = 64 if is_large else 32
        num_layers = 4 if is_large else 2

        model = SCNet(
            in_channels=2,
            num_sources=len(self.supported_stems),
            hidden_dim=hidden_dim,
            num_dprnn_layers=num_layers,
        )

        if self.checkpoint_path is not None:
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
            checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
            state_dict = (
                checkpoint.get("state_dict")
                or checkpoint.get("model")
                or checkpoint
            )
            model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()
        self._model = model

    def unload(self) -> None:
        """Drop model weights reference and purge cache."""
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _process_chunks(self, wav: torch.Tensor, segment_len: int = 44100 * 5, overlap: float = 0.25) -> torch.Tensor:
        """Process stereo audio in overlapping windows with Hann crossfading."""
        assert self._model is not None
        B, C, T = wav.shape
        if T <= segment_len:
            return self._model(wav)

        step = max(1, int(segment_len * (1.0 - overlap)))
        out = torch.zeros(B, len(self.supported_stems), C, T, device=wav.device)
        weight = torch.zeros(T, device=wav.device)
        window = torch.hann_window(segment_len, device=wav.device)

        for start in range(0, T, step):
            end = min(start + segment_len, T)
            actual_len = end - start
            chunk = wav[:, :, start:end]
            if actual_len < segment_len:
                pad = segment_len - actual_len
                chunk = F.pad(chunk, (0, pad))
            chunk_out = self._model(chunk)[:, :, :, :actual_len]
            w = window[:actual_len]
            out[:, :, :, start:end] += chunk_out * w
            weight[start:end] += w
            if end >= T:
                break

        weight = torch.clamp(weight, min=1e-6)
        out = out / weight.view(1, 1, 1, -1)
        return out

    def separate(
        self,
        input_audio: Union[Path, str],
        output_dir: Union[Path, str],
        shifts: Optional[int] = None,
        overlap: Optional[float] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """Separate audio into 4 stems and write 44.1kHz 16-bit PCM WAV files."""
        in_p = Path(input_audio).resolve()
        out_d = Path(output_dir).resolve()
        out_d.mkdir(parents=True, exist_ok=True)

        if not in_p.is_file():
            raise FileNotFoundError(f"Input file not found: {in_p}")

        effective_shifts = shifts if shifts is not None else self.shifts
        effective_overlap = overlap if overlap is not None else self.overlap

        # Read input audio
        audio_data, sr = sf.read(str(in_p), dtype="float32", always_2d=True)
        if audio_data.shape[0] == 0:
            raise ValueError(f"Input audio file is empty or zero-length: {in_p}")

        # Convert to tensor [C, T]
        wav_tensor = torch.from_numpy(audio_data.T)
        if sr != 44100:
            wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 44100)

        # Ensure stereo
        if wav_tensor.shape[0] == 1:
            wav_tensor = wav_tensor.repeat(2, 1)
        elif wav_tensor.shape[0] > 2:
            wav_tensor = wav_tensor[:2]

        completion_marker = out_d / f".complete_{self.model_name}"

        try:
            # Handle silent audio
            if wav_tensor.abs().max() < 1e-5:
                if progress_callback:
                    progress_callback("Audio is silent; generating zero-signal stems.")
                zero_sig = torch.zeros_like(wav_tensor).cpu().numpy().T
                results = {}
                for name in self.supported_stems:
                    target_path = out_d / f"{name}.wav"
                    tmp_path = out_d / f"{name}_tmp.wav"
                    sf.write(str(tmp_path), zero_sig, 44100, subtype="PCM_16")
                    tmp_path.replace(target_path)
                    results[name] = target_path
                completion_marker.write_text("ok")
                return results

            if progress_callback:
                progress_callback(f"Running SCNet ({self.model_name}) on device '{self.device}'...")

            self.load()
            assert self._model is not None

            # Input normalization
            ref = wav_tensor.mean(0)
            std = ref.std()
            if std < 1e-8:
                std = torch.tensor(1.0)
            wav_norm = (wav_tensor - ref.mean()) / std

            wav_batch = wav_norm.unsqueeze(0).to(self.device)  # [1, 2, T]
            T_samples = wav_batch.shape[-1]

            with torch.no_grad():
                if effective_shifts <= 1:
                    sources = self._process_chunks(wav_batch, overlap=effective_overlap)
                else:
                    sources = torch.zeros(
                        1, len(self.supported_stems), 2, T_samples, device=self.device
                    )
                    for shift_idx in range(effective_shifts):
                        shift_amount = int(shift_idx * (44100 // max(effective_shifts, 2)))
                        shifted_input = torch.roll(wav_batch, shifts=shift_amount, dims=-1)
                        shifted_out = self._process_chunks(shifted_input, overlap=effective_overlap)
                        sources += torch.roll(shifted_out, shifts=-shift_amount, dims=-1)
                    sources /= effective_shifts

                # Rescale back
                sources = sources[0] * std.to(self.device) + ref.mean().to(self.device)

            results = {}
            for name, source_tensor in zip(self.supported_stems, sources):
                target_path = out_d / f"{name}.wav"
                tmp_path = out_d / f"{name}_tmp.wav"
                stem_data = source_tensor.cpu().numpy().T  # [T, C]
                sf.write(str(tmp_path), stem_data, 44100, subtype="PCM_16")
                tmp_path.replace(target_path)
                results[name] = target_path

            completion_marker.write_text("ok")
            return results

        except Exception:
            if completion_marker.exists():
                completion_marker.unlink()
            for tmp_file in out_d.glob("*_tmp.wav"):
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            raise
