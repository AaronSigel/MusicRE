import asyncio
import json

import pytest
from typer.testing import CliRunner

import music_re.cli as cli
import music_re.mcp_server as server
from test_shared_pipeline import pipeline_env


runner = CliRunner()


@pytest.mark.parametrize(('args', 'expected'), [([], 'off'), (['--spectral-validation', 'audit'], 'audit'),
                                                (['--spectral-validation', 'FILTER'], 'filter')])
def test_cli_passes_spectral_mode_and_json_fields(monkeypatch, tmp_path, args, expected):
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return {'workspace_dir': str(tmp_path), 'spectral_validation': {'mode': kwargs['spectral_validation']},
                'active_midis': {}}

    monkeypatch.setattr(cli, 'run_pipeline', fake_pipeline)
    result = runner.invoke(cli.app, ['deconstruct', str(tmp_path / 'audio.wav'), '--json', *args])
    assert result.exit_code == 0, result.output
    assert captured['spectral_validation'] == expected
    assert json.loads(result.stdout)['spectral_validation']['mode'] == expected


def test_cli_json_includes_pipeline_spectral_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, 'run_pipeline', lambda *a, **kw: {
        'workspace_dir': str(tmp_path), 'spectral_validation': {'mode': kw['spectral_validation']},
        'active_midis': {'vocals': 'v.mid'},
    })
    result = runner.invoke(cli.app, ['deconstruct', str(tmp_path / 'audio.wav'), '--json'])
    assert result.exit_code == 0
    assert json.loads(result.stdout)['spectral_validation'] == {'mode': 'off'}
    assert json.loads(result.stdout)['active_midis'] == {'vocals': 'v.mid'}


def test_cli_rejects_invalid_spectral_mode(monkeypatch, tmp_path):
    called = False

    def fake_pipeline(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, 'run_pipeline', fake_pipeline)
    result = runner.invoke(cli.app, ['deconstruct', str(tmp_path / 'audio.wav'), '--spectral-validation', 'invalid'])
    assert result.exit_code != 0
    assert not called


def test_cli_help_lists_spectral_option():
    result = runner.invoke(cli.app, ['deconstruct', '--help'])
    assert '--spectral-validation' in result.stdout


def test_cli_json_rejects_nan(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, 'run_pipeline', lambda *a, **kw: {'value': float('nan')})
    result = runner.invoke(cli.app, ['deconstruct', str(tmp_path / 'audio.wav'), '--json'])
    assert result.exit_code == 1


@pytest.mark.parametrize(('mode', 'expected'), [('default', 'off'), ('filter', 'filter')])
def test_mcp_passes_spectral_mode(monkeypatch, tmp_path, mode, expected):
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return {'spectral_validation': {'mode': kwargs['spectral_validation']}}

    from contextlib import nullcontext
    monkeypatch.setattr(server, 'WorkspaceManager', lambda **kwargs: type('WS', (), {'lock': lambda self, p: nullcontext()})())
    monkeypatch.setattr(server, 'run_pipeline', fake_pipeline)
    kwargs = {} if mode == 'default' else {'spectral_validation': mode}
    asyncio.run(server.deconstruct_full_pipeline(str(tmp_path / 'audio.wav'), **kwargs))
    assert captured['spectral_validation'] == expected


def test_mcp_invalid_mode_raises_before_writing_manifest(tmp_path):
    source = tmp_path / 'audio.wav'
    source.write_bytes(b'')
    cache = tmp_path / 'cache'
    with pytest.raises(ValueError, match='spectral_validation'):
        asyncio.run(server.deconstruct_full_pipeline(str(source), cache_root=str(cache), spectral_validation='invalid'))
    assert not list(cache.rglob('manifest.json'))


def test_mcp_uppercase_mode_runs_like_cli(pipeline_env):
    pipeline, ws, source, _ = pipeline_env
    result = asyncio.run(server.deconstruct_full_pipeline(str(source), cache_root=str(ws.cache_root),
                                                          spectral_validation='AUDIT'))
    assert result['spectral_validation']['mode'] == 'audit'
    assert ws.get_manifest(source).spectral_validation['mode'] == 'audit'
