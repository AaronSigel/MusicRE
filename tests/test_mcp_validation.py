import asyncio
import time
from types import SimpleNamespace

import pytest

import music_re.mcp_server as server
from music_re.core import backends


def test_transcribe_stem_missing_timeline_raises(tmp_path, monkeypatch):
    stem = tmp_path / "bass.wav"
    stem.write_bytes(b"")
    missing = tmp_path / "nope" / "timeline.json"
    called = []
    monkeypatch.setattr(
        server, "get_transcriber_backend",
        lambda **kw: SimpleNamespace(transcribe=lambda **k: called.append(k)),
    )
    with pytest.raises(FileNotFoundError, match="timeline.json"):
        asyncio.run(server.transcribe_stem(str(stem), timeline_path=str(missing)))
    assert called == []


def test_get_transcriber_backend_rejects_unknown_mode():
    with pytest.raises(ValueError) as exc:
        backends.get_transcriber_backend(mode="drmus")
    msg = str(exc.value)
    assert "drmus" in msg and "tonal" in msg and "drums" in msg


def test_tool_does_not_block_event_loop(tmp_path, monkeypatch):
    def slow_analyze(p):
        time.sleep(0.3)
        return SimpleNamespace(model_dump=lambda: {"ok": True})

    monkeypatch.setattr(server, "core_analyze_audio", slow_analyze)

    async def main():
        gaps = []

        async def ticker():
            last = time.perf_counter()
            for _ in range(20):
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        tick_task = asyncio.create_task(ticker())
        await asyncio.sleep(0.02)
        result = await server.analyze_track(str(tmp_path / "x.wav"))
        await tick_task
        return result, gaps

    result, gaps = asyncio.run(main())
    assert result == {"ok": True}
    assert max(gaps) < 0.1


@pytest.mark.parametrize("tool", ["separate_stems", "get_structure_timeline", "evaluate_quality_report"])
def test_cache_root_reaches_workspace_manager(tmp_path, monkeypatch, tool):
    seen = []

    class FakeWS:
        def __init__(self, cache_root=None):
            seen.append(cache_root)

        def get_manifest(self, p):
            return SimpleNamespace(stems={})

        def save_timeline(self, p, t):
            pass

        def save_quality_report(self, p, r):
            pass

    dump = SimpleNamespace(model_dump=lambda: {})
    monkeypatch.setattr(server, "WorkspaceManager", FakeWS)
    monkeypatch.setattr(server, "core_separate_stems", lambda *a, **k: dump)
    monkeypatch.setattr(server, "core_evaluate_quality", lambda **k: dump)
    monkeypatch.setattr(
        server, "get_structure_analyzer",
        lambda **k: SimpleNamespace(analyze=lambda *a, **kw: dump),
    )
    cache = str(tmp_path / "cache")
    asyncio.run(getattr(server, tool)(str(tmp_path / "x.wav"), cache_root=cache))
    assert seen == [cache]


def test_structure_backend_list_independent_of_call_order(monkeypatch):
    monkeypatch.setattr(backends, "_STRUCTURE_REGISTRY", {})
    before = backends.list_structure_backends()
    try:
        backends.get_structure_analyzer(name="librosa")
    except Exception:
        pass
    after = backends.list_structure_backends()
    assert before == after
