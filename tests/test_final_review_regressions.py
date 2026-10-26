import numpy as np
import pretty_midi
import soundfile as sf

from music_re.core.midi_quantize import quantize_to_timeline
from music_re.core.separator import separate_stems
from music_re.workspace import WorkspaceManager


def test_profile_return_does_not_reuse_corrupt_separation(tmp_path):
    ws = WorkspaceManager(tmp_path / 'cache')
    audio = tmp_path / 'silence.wav'
    sf.write(audio, np.zeros((44100 * 2, 2)), 44100)
    first = separate_stems(audio, ws, shifts=0, overlap=.1, device='cpu')
    with open(first.stems['vocals'], 'wb') as stream:
        stream.write(b'nonempty corrupt cache')
    separate_stems(audio, ws, shifts=1, overlap=.25, device='cpu')
    final = separate_stems(audio, ws, shifts=0, overlap=.1, device='cpu')
    assert sf.info(final.stems['vocals']).frames == 88200


def test_standalone_regeneration_invalidates_same_path_dependents(tmp_path):
    from pathlib import Path
    ws = WorkspaceManager(tmp_path / 'cache')
    audio = tmp_path / 'silence.wav'
    sf.write(audio, np.zeros(44100), 44100)
    first = separate_stems(audio, ws, device='cpu')
    ws.mark_stage_complete(audio, 'structure', {'timeline': {'bpm': 120}})
    ws.mark_stage_complete(audio, 'transcription', {'vocals': '/old.mid'})
    ws.mark_stage_complete(audio, 'evaluation', {'report': {}})
    ws.update_manifest(audio, status='completed')
    Path(first.stems['vocals']).unlink()
    separate_stems(audio, ws, device='cpu')
    manifest = ws.get_manifest(audio)
    assert manifest.midis == {}
    assert manifest.timeline is None
    assert manifest.evaluation is None
    assert manifest.completed_stages == ['separation']
    assert manifest.status != 'completed'


def test_onset_only_preserves_short_positive_remaining_duration():
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    instrument.notes.append(pretty_midi.Note(100, 60, .07, .16))
    pm.instruments.append(instrument)
    note = quantize_to_timeline(pm, quantize_ends=False).instruments[0].notes[0]
    assert note.start == .125
    assert note.end == .16
