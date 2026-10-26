import asyncio
from music_re.mcp_server import mcp

def test_mcp_tools_registered():
    tools = asyncio.run(mcp.list_tools())
    tool_names = [t.name for t in tools]
    assert "slice_audio" in tool_names
    assert "analyze_track" in tool_names
    assert "separate_stems" in tool_names
    assert "transcribe_stem" in tool_names
    assert "inspect_midi_bars" in tool_names
    assert "deconstruct_full_pipeline" in tool_names
    assert "get_structure_timeline" in tool_names
    assert "evaluate_quality_report" in tool_names
    assert "export_project_bundle" in tool_names
