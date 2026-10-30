import contextlib
import gc
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Union
import soundfile as sf
import torch

from music_re.core.backends.separator_base import SeparatorBackend

STEMS_4 = ["vocals", "drums", "bass", "other"]
STEMS_6 = ["vocals", "drums", "bass", "other", "guitar", "piano"]


class DemucsSeparator(SeparatorBackend):
    """Adapter wrapping Facebook Demucs model with device and memory management."""

    def __init__(
        self,
        six_stems: bool = False,
        device: str = "auto",
        **kwargs,
    ):
        model_name = "htdemucs_6s" if six_stems else "htdemucs"
        super().__init__(backend_name="demucs", model_name=model_name, device=device)
        self.six_stems = six_stems
        self.shifts = kwargs.get('shifts', 1)
        self.overlap = kwargs.get('overlap', .25)
        self._model = None
        self.quality_note = (
            "6-stem mode (htdemucs_6s) carries noticeable acoustic bleed between guitar and piano."
            if six_stems
            else "4-stem mode (htdemucs) provides optimal source isolation."
        )

    @property
    def supported_stems(self) -> List[str]:
        return list(STEMS_6 if self.six_stems else STEMS_4)

    def load(self) -> None:
        """Load pretrained Demucs model weights if not already loaded."""
        if self._model is not None:
            return

        from demucs_infer.pretrained import get_model
        from demucs_infer.states import load_model

        with contextlib.redirect_stdout(sys.stderr):
            try:
                # Direct load from cache if pre-packaged checkpoint exists
                cache_file = Path.home() / ".cache" / "demucs-infer" / f"5c90dfd2-34c22ccb.th"
                if self.model_name == "htdemucs_6s" and cache_file.is_file():
                    model = load_model(cache_file)
                else:
                    model = get_model(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Demucs model '{self.model_name}': {e}"
                ) from e

            model.to(self.device)
            model.eval()
            self._model = model

    def unload(self) -> None:
        """Drop model weights reference and purge cache."""
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def separate(
        self,
        input_audio: Union[Path, str],
        output_dir: Union[Path, str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Path]:
        """Separate audio into stems, writing output WAV files atomically."""
        in_p = Path(input_audio).resolve()
        out_d = Path(output_dir).resolve()
        out_d.mkdir(parents=True, exist_ok=True)

        if not in_p.is_file():
            raise FileNotFoundError(f"Input file not found: {in_p}")

        completion_marker = out_d / f".complete_{self.model_name}"

        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio

        # Read audio
        wav_raw = AudioFile(str(in_p)).read(
            streams=0,
            samplerate=44100,
            channels=2,
        )

        try:
            # Handle silent audio
            if wav_raw.abs().max() < 1e-5:
                if progress_callback:
                    progress_callback("Audio is silent; generating zero-signal stems.")
                zero_sig = torch.zeros_like(wav_raw)
                results = {}
                for name in self.supported_stems:
                    target_path = out_d / f"{name}.wav"
                    tmp_path = out_d / f"{name}_tmp.wav"
                    save_audio(zero_sig, str(tmp_path), samplerate=44100)
                    tmp_path.replace(target_path)
                    results[name] = target_path
                completion_marker.write_text("ok")
                return results

            if progress_callback:
                progress_callback(f"Running Demucs ({self.model_name}) on device '{self.device}'...")

            self.load()
            assert self._model is not None

            ref = wav_raw.mean(0)
            std = ref.std()
            if std < 1e-8:
                std = torch.tensor(1.0)
            wav = (wav_raw - ref.mean()) / std

            with torch.no_grad():
                sources = apply_model(self._model, wav[None].to(self.device), shifts=self.shifts,
                                      overlap=self.overlap, device=self.device)[0]
                sources = sources * std.to(self.device) + ref.mean().to(self.device)

            results = {}
            for name, source in zip(self._model.sources, sources):
                target_path = out_d / f"{name}.wav"
                tmp_path = out_d / f"{name}_tmp.wav"
                save_audio(source.cpu(), str(tmp_path), samplerate=self._model.samplerate)
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
