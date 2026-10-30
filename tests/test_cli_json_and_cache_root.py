import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import music_re.cli as cli
from music_re.cli import app
from music_re.schemas import (MidiDigest, MusicalTimeline, NoteEvent, QualityReport,
                              StemSeparationResult, TrackAnalysis)
from music_re.workspace import WorkspaceManager

runner = CliRunner()


def _hostile_file(tmp_path: Path, suffix: str) -> Path:
    # '[/red]' contains a slash, so the markup spans path components on purpose.
    f = Path(f'{tmp_path}/каталог_{"x" * 60}/[red]x[/red]_{"y" * 60}/трек_{"z" * 40}/'
             f'песня [bold]{"w" * 30}[/bold]{suffix}')
    f.parent.mkdir(parents=True)
    f.write_bytes(b'not really audio')
    assert len(str(f)) >= 200 and '[red]x[/red]' in str(f)
    return f


def _assert_json(result, expected_path: Path):
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert str(expected_path) in json.dumps(data, ensure_ascii=False)
    return data


def test_analyze_json_is_plain(tmp_path, monkeypatch):
    f = _hostile_file(tmp_path, '.wav')
    monkeypatch.setattr(cli, 'analyze_audio', lambda p: TrackAnalysis(
        file_path=str(p), duration_sec=1.0, bpm=120.0, key='C major', key_confidence=0.5,
        lufs=-14.0, beat_times=[0.0, 0.5]))
    _assert_json(runner.invoke(app, ['analyze', str(f), '--json']), f)


class _FakeAnalyzer:
    backend_name = 'fake'

    def analyze(self, file, stems=None, device='auto'):
        return MusicalTimeline(duration_sec=1.0, bpm=120.0, beats=[0.0], downbeats=[0.0],
                               backend='fake', warnings=[f'source {file}'])


def test_structure_json_is_plain_and_uses_cache_root(tmp_path, monkeypatch):
    f = _hostile_file(tmp_path, '.wav')
    cache = tmp_path / 'cache'
    monkeypatch.setattr(cli, 'get_structure_analyzer', lambda name: _FakeAnalyzer())
    result = runner.invoke(app, ['structure', str(f), '--json', '--cache-root', str(cache)])
    _assert_json(result, f)
    assert WorkspaceManager(cache_root=cache).get_timeline_path(f).exists()


def test_inspect_midi_json_is_plain(tmp_path, monkeypatch):
    f = _hostile_file(tmp_path, '.mid')
    monkeypatch.setattr(cli, 'generate_midi_digest', lambda p, bpm, start_bar, end_bar: MidiDigest(
        midi_path=str(p), bpm=bpm, instrument_type='tonal', total_notes=1, bars_analyzed='1-8',
        notes=[NoteEvent(note='C4', pitch=60, bar=1, beat_in_bar=1.0, duration_beats=1.0, velocity=90)]))
    _assert_json(runner.invoke(app, ['inspect-midi', str(f), '--json']), f)


def _fake_report(ws, file, gt_stems=None, gt_midi=None):
    return QualityReport(is_objective=False,
                         separation_metrics={'mode': 'diagnostic', 'source': str(file)},
                         transcription_metrics={'mode': 'diagnostic'},
                         audio_diagnostics={}, evaluated_at='2026-10-30T00:00:00+00:00')


def test_evaluate_json_is_plain_and_uses_cache_root(tmp_path, monkeypatch):
    f = _hostile_file(tmp_path, '.wav')
    cache = tmp_path / 'cache'
    ws = WorkspaceManager(cache_root=cache)
    ws.update_manifest(f, stems={'vocals': str(tmp_path / 'vocals.wav')})
    seen = {}

    def fake(ws_arg, file, gt_stems=None, gt_midi=None):
        seen['cache_root'] = ws_arg.cache_root
        return _fake_report(ws_arg, file)

    monkeypatch.setattr(cli, 'evaluate_quality', fake)
    _assert_json(runner.invoke(app, ['evaluate', str(f), '--json', '--cache-root', str(cache)]), f)
    assert seen['cache_root'] == cache.resolve()
    assert ws.get_quality_report_path(f).exists()


def test_evaluate_empty_workspace_fails_naming_cache_root(tmp_path, monkeypatch):
    f = tmp_path / 'song.wav'
    f.write_bytes(b'x')
    cache = tmp_path / 'empty_cache'
    called = []
    monkeypatch.setattr(cli, 'evaluate_quality', lambda *a, **k: called.append(1) or _fake_report(*a))
    result = runner.invoke(app, ['evaluate', str(f), '--json', '--cache-root', str(cache)])
    assert result.exit_code != 0
    assert not called
    assert str(cache.resolve()) in result.output.replace('\n', '')
    assert 'stems_count' not in result.output


def test_split_uses_cache_root(tmp_path, monkeypatch):
    f = tmp_path / 'song.wav'
    f.write_bytes(b'x')
    cache = tmp_path / 'cache'
    seen = {}

    def fake_separate(file, workspace, six_stems, device, progress_callback):
        seen['cache_root'] = workspace.cache_root
        return StemSeparationResult(workspace_dir=str(workspace.get_track_dir(file)),
                                    model_used='fake', stems={})

    monkeypatch.setattr(cli, 'separate_stems', fake_separate)
    result = runner.invoke(app, ['split', str(f), '--cache-root', str(cache)])
    assert result.exit_code == 0, result.output
    assert seen['cache_root'] == cache.resolve()


@pytest.mark.parametrize('cmd', ['split', 'structure', 'evaluate'])
def test_cache_root_option_in_help(cmd):
    result = runner.invoke(app, [cmd, '--help'])
    assert result.exit_code == 0
    assert '--cache-root' in result.stdout
