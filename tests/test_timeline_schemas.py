import pytest
from music_re.schemas import TimelineSection, MusicalTimeline, WorkspaceManifest

def test_timeline_section_schema():
    sec = TimelineSection(label="verse", start_sec=8.0, end_sec=24.0, confidence=0.85)
    assert sec.label == "verse"
    assert sec.start_sec == 8.0
    assert sec.end_sec == 24.0
    assert sec.confidence == 0.85

def test_musical_timeline_schema():
    tl = MusicalTimeline(
        duration_sec=32.0,
        bpm=120.0,
        beats=[0.0, 0.5, 1.0, 1.5, 2.0],
        downbeats=[0.0, 2.0],
        sections=[
            TimelineSection(label="intro", start_sec=0.0, end_sec=8.0, confidence=0.9),
            TimelineSection(label="verse", start_sec=8.0, end_sec=32.0, confidence=0.8),
        ],
    )
    assert tl.bpm == 120.0
    assert len(tl.beats) == 5
    assert len(tl.downbeats) == 2
    assert len(tl.sections) == 2

    # JSON roundtrip
    data = tl.model_dump()
    tl_loaded = MusicalTimeline(**data)
    assert tl_loaded.bpm == tl.bpm
    assert tl_loaded.sections[0].label == "intro"

def test_workspace_manifest_timeline_field():
    m = WorkspaceManifest(track_id="abc123", input_file="/path/to/song.wav")
    assert hasattr(m, "timeline")
    assert m.timeline is None
