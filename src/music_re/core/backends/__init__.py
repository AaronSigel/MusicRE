from typing import Dict, List, Type
from music_re.core.backends.separator_base import SeparatorBackend, resolve_device

_SEPARATOR_REGISTRY: Dict[str, Type[SeparatorBackend]] = {}


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
