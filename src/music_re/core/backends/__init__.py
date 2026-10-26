from typing import Dict, List, Type
from music_re.core.backends.separator_base import SeparatorBackend, resolve_device
from music_re.core.backends.structure_base import StructureAnalyzerBackend

_SEPARATOR_REGISTRY: Dict[str, Type[SeparatorBackend]] = {}
_STRUCTURE_REGISTRY: Dict[str, Type[StructureAnalyzerBackend]] = {}


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
            import allin1
            return _STRUCTURE_REGISTRY["allin1"](**kwargs)
        except (ImportError, KeyError):
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
