import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from music_re.core.backends.allin1_backend import AllInOneStructureAnalyzer


def test_allin1_uses_infer_api_and_keeps_stems(tmp_path, monkeypatch):
    audio = tmp_path / 'song.wav'
    sf.write(audio, np.zeros(44100 * 3), 44100)
    stems = {}
    for name in ('bass', 'drums', 'vocals', 'other', 'guitar', 'piano'):
        path = tmp_path / f'{name}.wav'
        sf.write(path, np.full((44100 * 3, 2), 0.1), 44100, subtype='FLOAT')
        stems[name] = str(path)

    class StemsInput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def analyze(*, stems_input, device, model, demix_dir, spec_dir, keep_byproducts, multiprocess):
        supplied = stems_input[0]
        other, _ = sf.read(supplied.other)
        assert other[0, 0] == pytest.approx(0.3)
        assert model == 'harmonix-all'
        return [SimpleNamespace(bpm=120, beats=[0.5, 1.0], downbeats=[0.5],
                                segments=[SimpleNamespace(label='intro', start=0, end=2)])]

    monkeypatch.setitem(sys.modules, 'allin1_infer', SimpleNamespace(analyze=analyze, StemsInput=StemsInput))
    timeline = AllInOneStructureAnalyzer().analyze(audio, stems=stems, device='cpu')
    assert timeline.duration_sec == 3
    assert timeline.sections[0].confidence is None
    assert timeline.backend == 'allin1'
    assert all(sf.info(p).frames == 132300 for p in stems.values())


def test_allin1_rejects_empty_result(tmp_path, monkeypatch):
    audio = tmp_path / 'song.wav'
    sf.write(audio, np.zeros(44100), 44100)
    monkeypatch.setitem(sys.modules, 'allin1_infer', SimpleNamespace(analyze=lambda **kwargs: []))
    with pytest.raises(RuntimeError, match='no result'):
        AllInOneStructureAnalyzer().analyze(audio, device='cpu')
