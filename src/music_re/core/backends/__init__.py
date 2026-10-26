from typing import Dict, List, Type
from music_re.core.backends.separator_base import SeparatorBackend, resolve_device
from music_re.core.backends.structure_base import StructureAnalyzerBackend
from music_re.core.backends.transcriber_base import TranscriberBackend

_SEPARATOR_REGISTRY: Dict[str, Type[SeparatorBackend]] = {}
_STRUCTURE_REGISTRY: Dict[str, Type[StructureAnalyzerBackend]] = {}
_TRANSCRIBER_REGISTRY: Dict[str, Type[TranscriberBackend]] = {}


def register_separator_backend(name: str, backend_cls: Type[SeparatorBackend]) -> None:
    """Register a new separator backend class."""
    _SEPARATOR_REGISTRY[name.lower()] = backend_cls


def get_separator_backend(name: str = "demucs", **kwargs) -> SeparatorBackend:
    """Instantiate a separator backend by name."""
    # Lazy register default backends if not already registered
    if "demucs" not in _SEPARATOR_REGISTRY:
        try:
            from music_re.core.backends.demucs_backend import DemucsSeparator
            register_separator_backend("demucs", DemucsSeparator)
        except ImportError:
            pass

    key = name.lower()
    if key not in _SEPARATOR_REGISTRY:
        available = ", ".join(sorted(_SEPARATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown separator backend '{name}'. Available backends: {available or 'none'}"
        )
    return _SEPARATOR_REGISTRY[key](**kwargs)


def list_separator_backends() -> List[str]:
    """List names of registered separator backends."""
    if "demucs" not in _SEPARATOR_REGISTRY:
        try:
            from music_re.core.backends.demucs_backend import DemucsSeparator
            register_separator_backend("demucs", DemucsSeparator)
        except ImportError:
            pass
    return sorted(_SEPARATOR_REGISTRY.keys())


def register_structure_analyzer(name: str, backend_cls: Type[StructureAnalyzerBackend]) -> None:
    """Register a new structure analyzer backend class."""
    _STRUCTURE_REGISTRY[name.lower()] = backend_cls


def get_structure_analyzer(name: str = "auto", **kwargs) -> StructureAnalyzerBackend:
    """Instantiate a structure analyzer backend by name, with auto fallback."""
    # Lazy register default backends
    if "librosa" not in _STRUCTURE_REGISTRY:
        try:
            from music_re.core.backends.librosa_structure_backend import LibrosaStructureAnalyzer
            register_structure_analyzer("librosa", LibrosaStructureAnalyzer)
        except ImportError:
            pass

    if "allin1" not in _STRUCTURE_REGISTRY:
        try:
            from music_re.core.backends.allin1_backend import AllInOneStructureAnalyzer
            register_structure_analyzer("allin1", AllInOneStructureAnalyzer)
        except ImportError:
            pass

    key = name.lower()
    if key == "auto":
        # Prefer All-In-One if installed, fallback to Librosa
        try:
            import allin1_infer
            return _STRUCTURE_REGISTRY["allin1"](**kwargs)
        except ModuleNotFoundError as e:
            if e.name != 'allin1_infer':
                raise
            return _STRUCTURE_REGISTRY["librosa"](**kwargs)

    if key not in _STRUCTURE_REGISTRY:
        available = ", ".join(sorted(_STRUCTURE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown structure analyzer backend '{name}'. Available backends: {available or 'none'}"
        )
    return _STRUCTURE_REGISTRY[key](**kwargs)


def list_structure_backends() -> List[str]:
    """List names of registered structure analyzer backends."""
    if "librosa" not in _STRUCTURE_REGISTRY:
        try:
            from music_re.core.backends.librosa_structure_backend import LibrosaStructureAnalyzer
            register_structure_analyzer("librosa", LibrosaStructureAnalyzer)
        except ImportError:
            pass
    return sorted(_STRUCTURE_REGISTRY.keys())


def register_transcriber_backend(name: str, backend_cls: Type[TranscriberBackend]) -> None:
    """Register a new transcriber backend class."""
    _TRANSCRIBER_REGISTRY[name.lower()] = backend_cls


def get_transcriber_backend(
    mode: str = "tonal",
    name: str = "auto",
    **kwargs,
) -> TranscriberBackend:
    """Instantiate a transcriber backend by mode and name."""
    # Lazy register default backends
    if "basic_pitch" not in _TRANSCRIBER_REGISTRY:
        try:
            from music_re.core.backends.basic_pitch_backend import BasicPitchTranscriber
            register_transcriber_backend("basic_pitch", BasicPitchTranscriber)
        except ImportError:
            pass

    if "drum_multiband" not in _TRANSCRIBER_REGISTRY:
        try:
            from music_re.core.backends.drum_backend import DrumTranscriber
            register_transcriber_backend("drum_multiband", DrumTranscriber)
        except ImportError:
            pass

    req_name = name.lower()
    if req_name == "auto":
        if mode == "drums":
            req_name = "drum_multiband"
        else:
            req_name = "basic_pitch"

    if req_name not in _TRANSCRIBER_REGISTRY:
        available = ", ".join(sorted(_TRANSCRIBER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown transcriber backend '{name}'. Available backends: {available or 'none'}"
        )

    return _TRANSCRIBER_REGISTRY[req_name](**kwargs)


def list_transcriber_backends() -> List[str]:
    """List names of registered transcriber backends."""
    if "basic_pitch" not in _TRANSCRIBER_REGISTRY:
        try:
            from music_re.core.backends.basic_pitch_backend import BasicPitchTranscriber
            register_transcriber_backend("basic_pitch", BasicPitchTranscriber)
        except ImportError:
            pass
    if "drum_multiband" not in _TRANSCRIBER_REGISTRY:
        try:
            from music_re.core.backends.drum_backend import DrumTranscriber
            register_transcriber_backend("drum_multiband", DrumTranscriber)
        except ImportError:
            pass
    return sorted(_TRANSCRIBER_REGISTRY.keys())
