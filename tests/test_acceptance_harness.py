import subprocess
import sys

import pytest


def test_runner_does_not_hide_subprocess_failure(tmp_path):
    from scripts.verify_real_e2e import run_command
    with pytest.raises(subprocess.CalledProcessError):
        run_command([sys.executable, '-c', "raise RuntimeError('failed inference')"], tmp_path / 'run.log')
    assert 'failed inference' in (tmp_path / 'run.log').read_text()


def test_acceptance_rejects_incomplete_export(tmp_path):
    from scripts.verify_real_e2e import inspect_export
    with pytest.raises(ValueError, match='manifest'):
        inspect_export(tmp_path)


def test_mcp_structured_result_wrapper_is_unwrapped():
    from mcp.types import CallToolResult, TextContent
    from scripts.verify_real_e2e import tool_payload
    response = CallToolResult(content=[TextContent(type='text', text='{"workspace_dir":"/cache"}')],
                              structuredContent={'result': {'workspace_dir': '/cache'}})
    assert tool_payload(response) == {'workspace_dir': '/cache'}
