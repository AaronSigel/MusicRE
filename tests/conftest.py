from pathlib import Path
from shutil import copyfile

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from music_re.core.profiles import resolve_profile
from music_re.schemas import MusicalTimeline
from music_re.workspace import WorkspaceManager
from music_re.core.quality import evaluate_quality


@pytest.fixture
def seed_stem_cache():
    """Register synthetic stems as a valid cache fixture; never load ML weights."""
    from music_re.core.backends import get_separator_backend
    from music_re.core.separation_cache import cache_context, save_cache_record
    def seed(source, stems, six_stems=False, shifts=1, overlap=.25):
        backend = get_separator_backend(six_stems=six_stems, device='cpu', shifts=shifts, overlap=overlap)
        directory = Path(next(iter(stems.values()))).parent
        save_cache_record(directory / f'.complete_{backend.model_name}', stems,
                          cache_context(source, backend, shifts, overlap))
    return seed


@pytest.fixture
def complete_workspace(tmp_path):
    """A complete registered bundle, not a model-accuracy fixture."""
    ws = WorkspaceManager(tmp_path / 'complete-cache')
    source = tmp_path / 'complete.wav'
    sf.write(source, np.sin(np.arange(22050) * .1) * .2, 22050)
    stems, midis, results = {}, {}, {}
    root = ws.get_track_dir(source)
    (root / 'midis').mkdir()
    for name in ['vocals', 'drums', 'bass', 'other']:
        stem = ws.get_stems_dir(source, 'fixture') / f'{name}.wav'
        sf.write(stem, np.zeros(22050), 22050)
        stems[name] = str(stem)
        mid = root / 'midis' / f'{name}.mid'
        pretty_midi.PrettyMIDI().write(str(mid))
        raw = mid.with_name(f'{name}_raw.mid')
        copyfile(mid, raw)
        midis[name] = str(mid)
        results[name] = dict(status='no_notes', note_count=0, backend='fixture',
                             midi_path=str(mid), raw_midi_path=str(raw))
    combined = root / 'midis' / 'all_stems.mid'
    pretty_midi.PrettyMIDI().write(str(combined))
    midis['all_stems'] = str(combined)
    ws.mark_stage_complete(source, 'separation', stems)
    ws.save_metadata(source, {'bpm': 120})
    ws.mark_stage_complete(source, 'analysis', {'bpm': 120})
    ws.save_timeline(source, MusicalTimeline(duration_sec=1, bpm=120, beats=[0, .5], downbeats=[]))
    ws.mark_stage_complete(source, 'transcription', midis)
    ws.save_quality_report(source, evaluate_quality(ws, source))
    ws.update_manifest(source, status='completed', transcription_results=results,
                        run_config=resolve_profile('balanced').to_dict())
    return ws, source
