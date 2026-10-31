import asyncio

from test_shared_pipeline import pipeline_env


def test_mcp_matches_pipeline_result_in_filter_mode(pipeline_env, monkeypatch):
    import music_re.mcp_server as server

    pipeline, workspace, source, counts = pipeline_env
    monkeypatch.setattr(server, 'WorkspaceManager', lambda **kwargs: workspace)
    expected = pipeline.run_pipeline(source, workspace=workspace, profile='fast', spectral_validation='filter')
    calls = counts.copy()
    result = asyncio.run(server.deconstruct_full_pipeline(str(source), profile='fast', spectral_validation='filter'))
    assert result == expected
    assert counts == calls
