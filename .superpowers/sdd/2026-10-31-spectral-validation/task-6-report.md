# Task 6 report

## Outcome

Exposed `spectral_validation` through the `deconstruct` CLI command and MCP `deconstruct_full_pipeline` tool. CLI normalizes case and validates choices. CLI JSON returns the pipeline result, including `spectral_validation` and `active_midis`, with `allow_nan=False`.

No `validate_spectral_midi` tool was added. `separator_backend` MCP passthrough remains out of scope.

## Tests

- `uv run --locked pytest tests/test_spectral_cli_mcp.py -q`: 9 passed.
- `uv run --locked pytest -m "not models and not real_e2e and not dataset" -q`: 434 passed, 4 deselected.

## Concern

Installed Typer/Click path raises `BadParameter` directly for invalid CLI Choice values. `CliRunner` reports exit code 1, and direct `music-re` invocation prints traceback with status 1, rather than expected exit code 2. Choice rejects invalid values before pipeline call. Fix likely requires a wider Typer/Click compatibility change, outside this task's minimal scope.
