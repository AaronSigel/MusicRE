import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_shared_pipeline import pipeline_env
from test_spectral_pipeline_integration import stage_calls, tonal_pipeline_env


def test_mcp_matches_pipeline_result_in_filter_mode(pipeline_env, monkeypatch):
    import music_re.mcp_server as server

    pipeline, workspace, source, counts = pipeline_env
    monkeypatch.setattr(server, 'WorkspaceManager', lambda **kwargs: workspace)
    expected = pipeline.run_pipeline(source, workspace=workspace, profile='fast', spectral_validation='filter')
    calls = counts.copy()
    result = asyncio.run(server.deconstruct_full_pipeline(str(source), profile='fast', spectral_validation='filter'))
    assert result == expected
    assert counts == calls


SIX = ['vocals', 'drums', 'bass', 'other', 'guitar', 'piano']
TONES = {'bass': (110, 45), 'guitar': (220, 57), 'piano': (440, 69)}


@pytest.fixture
def six_stem_env(pipeline_env, monkeypatch):
    from music_re.core.backends.transcriber_base import save_transcription

    pipeline, ws, source, counts = pipeline_env
    separate = pipeline.separate_stems
    get_backend = pipeline.get_transcriber_backend

    def tonal_separate(*args, **kwargs):
        result = separate(*args, **kwargs)
        sr = 44100
        for name, (freq, _) in TONES.items():
            tone = .5 * np.sin(2 * np.pi * freq * np.arange(sr * 2) / sr)
            tone[sr:] = 0
            sf.write(result.stems[name], tone, sr)
        return result

    def backend(**kwargs):
        original = get_backend(**kwargs)

        def transcribe(audio_path, output_midi_path, **options):
            name = options['stem_name']
            if name not in TONES:
                return original.transcribe(audio_path, output_midi_path, **options)
            counts['transcription'] += 1
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            instrument.notes = [pretty_midi.Note(velocity=100, pitch=TONES[name][1], start=.1, end=.75),
                                pretty_midi.Note(velocity=100, pitch=80, start=1.25, end=1.75)]
            midi.instruments.append(instrument)
            return save_transcription(midi, output_midi_path, midi if options['save_raw'] else None, 'fake', {})

        return SimpleNamespace(transcribe=transcribe)

    monkeypatch.setattr(pipeline, 'separate_stems', tonal_separate)
    monkeypatch.setattr(pipeline, 'get_transcriber_backend', backend)
    return pipeline, ws, source, counts


def _report(ws, source):
    return json.loads(ws.get_spectral_validation_path(source).read_text())


def _pitches(path):
    return [n.pitch for i in pretty_midi.PrettyMIDI(str(path)).instruments for n in i.notes]


def test_six_stem_audit_scores_nothing_and_keeps_original(six_stem_env):
    pipeline, ws, source, _ = six_stem_env
    result = pipeline.run_pipeline(source, workspace=ws, six_stems=True, spectral_validation='audit')
    manifest = ws.get_manifest(source)
    assert set(manifest.stems) == set(SIX)
    assert set(manifest.spectral_validation['stems']) == set(SIX)
    assert result['active_midis'] == manifest.midis
    assert manifest.validated_midis == {}
    report = _report(ws, source)
    for name in SIX:
        stem = report['stems'][name]
        assert stem['status'] == ('skipped' if name == 'drums' else 'validated')
        assert all(note['decision'] == ('skipped' if name == 'drums' else 'unscored') for note in stem['notes'])
    assert len(report['stems']['piano']['notes']) == 2


def test_six_stem_filter_rejects_guitar_but_never_piano_or_drums(six_stem_env):
    pipeline, ws, source, counts = six_stem_env
    result = pipeline.run_pipeline(source, workspace=ws, six_stems=True, spectral_validation='filter')
    manifest = ws.get_manifest(source)
    assert set(manifest.validated_midis) == set(SIX) | {'all_stems'}
    assert set(result['active_midis']) == set(SIX) | {'all_stems'}
    stems = _report(ws, source)['stems']
    guitar = [note['decision'] for note in stems['guitar']['notes']]
    assert guitar[0] != 'reject' and guitar[1] == 'reject'
    assert _pitches(manifest.validated_midis['guitar']) == [57]
    assert _pitches(manifest.validated_midis['bass']) == [45]
    assert [note['decision'] for note in stems['piano']['notes']] == ['unscored', 'unscored']
    assert _pitches(manifest.validated_midis['piano']) == [69, 80]
    assert stems['drums']['status'] == 'skipped'
    assert all(note['decision'] == 'skipped' for note in stems['drums']['notes'])
    assert _pitches(manifest.validated_midis['drums']) == _pitches(manifest.midis['drums'])
    assert sorted(_pitches(manifest.validated_midis['all_stems'])) == sorted(
        p for name in SIX for p in _pitches(manifest.validated_midis[name]))
    assert counts['transcription'] == 6


@pytest.mark.parametrize('mode', ['audit', 'filter'])
def test_identical_rerun_reuses_every_stage(tonal_pipeline_env, stage_calls, monkeypatch, mode):
    pipeline, ws, source, counts = tonal_pipeline_env
    validate = pipeline.SpectralMidiValidator.validate
    validations = []

    def spy(self, *args, **kwargs):
        validations.append(args[2])
        return validate(self, *args, **kwargs)

    monkeypatch.setattr(pipeline.SpectralMidiValidator, 'validate', spy)
    first = pipeline.run_pipeline(source, workspace=ws, profile='fast', spectral_validation=mode)
    backend_calls, stages, validated = dict(counts), dict(stage_calls), list(validations)
    assert sorted(validated) == ['bass', 'drums', 'other', 'vocals']
    second = pipeline.run_pipeline(source, workspace=ws, profile='fast', spectral_validation=mode)
    assert counts == backend_calls
    assert stage_calls == stages
    assert validations == validated
    assert second == first


def test_filter_dawproject_note_counts_match_validated_midis(tonal_pipeline_env, tmp_path):
    from dawproject import DawProject
    from music_re.core.export import export_bundle
    pipeline, ws, source, _ = tonal_pipeline_env
    pipeline.run_pipeline(source, workspace=ws, spectral_validation='filter')
    manifest = ws.get_manifest(source)
    target = export_bundle(ws, source, tmp_path / 'bundle', export_dawproject=True)
    project = DawProject.load_project(str(target / 'song.dawproject'))
    for name, path in manifest.validated_midis.items():
        if name == 'all_stems':
            continue
        lanes = [lane for lane in project.arrangement.lanes.lanes if lane.track.name == f'{name} (MIDI)']
        exported = sum(len(clip.content.notes) for lane in lanes for clip in lane.clips)
        assert exported == len(_pitches(path)), name
    assert len(_pitches(manifest.midis['bass'])) == 2 and len(_pitches(manifest.validated_midis['bass'])) == 1


def test_spectral_validation_defaults_to_off_everywhere(pipeline_env, monkeypatch):
    import inspect
    import music_re.cli as cli
    import music_re.mcp_server as server
    from scripts import run_quality_benchmarks as bench
    pipeline, ws, source, _ = pipeline_env
    assert inspect.signature(pipeline.run_pipeline).parameters['spectral_validation'].default == 'off'
    assert bench.build_parser().parse_args([]).spectral_validation == ['off']
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs['spectral_validation'])
        return {'workspace_dir': str(ws.get_track_dir(source))}

    monkeypatch.setattr(cli, 'run_pipeline', spy)
    monkeypatch.setattr(server, 'run_pipeline', spy)
    monkeypatch.setattr(cli, 'WorkspaceManager', lambda **kwargs: ws)
    monkeypatch.setattr(server, 'WorkspaceManager', lambda **kwargs: ws)
    assert CliRunner().invoke(cli.app, ['deconstruct', str(source)]).exit_code == 0
    asyncio.run(server.deconstruct_full_pipeline(str(source)))
    assert seen == ['off', 'off']
