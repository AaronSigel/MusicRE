import pytest

from music_re.core.backends.structure_base import StructureAnalyzerBackend
from music_re.core.backends.librosa_structure_backend import LibrosaStructureAnalyzer
from music_re.core.backends.allin1_backend import AllInOneStructureAnalyzer
from music_re.core.backends import get_structure_analyzer, register_structure_analyzer

def test_structure_registry_librosa_lookup():
    analyzer = get_structure_analyzer(name="librosa")
    assert isinstance(analyzer, LibrosaStructureAnalyzer)
    assert analyzer.backend_name == "librosa"

def test_structure_registry_auto_lookup():
    analyzer = get_structure_analyzer(name="auto")
    assert isinstance(analyzer, StructureAnalyzerBackend)

def test_structure_registry_allin1_class():
    analyzer = AllInOneStructureAnalyzer()
    assert analyzer.backend_name == "allin1"

def test_structure_registry_unknown_raises():
    with pytest.raises(ValueError, match="Unknown structure analyzer backend"):
        get_structure_analyzer(name="nonexistent_analyzer")
