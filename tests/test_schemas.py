from music_re.schemas import TrackAnalysis, StemSeparationResult, NoteEvent, MidiDigest

def test_schemas_instantiation():
    analysis = TrackAnalysis(
        file_path="/tmp/test.wav",
        duration_sec=30.0,
        bpm=120.0,
        key="C major",
        key_confidence=0.85,
        lufs=-14.0,
        beat_times=[0.0, 0.5, 1.0, 1.5],
    )
    assert analysis.bpm == 120.0
    assert analysis.key == "C major"

    stems = StemSeparationResult(
        workspace_dir="/tmp/workspace",
        model_used="htdemucs",
        stems={"vocals": "/tmp/vocals.wav", "drums": "/tmp/drums.wav"},
    )
    assert len(stems.stems) == 2

    note = NoteEvent(
        note="C4",
        pitch=60,
        bar=1,
        beat_in_bar=1.0,
        duration_beats=1.0,
        velocity=90,
    )
    assert note.pitch == 60

    digest = MidiDigest(
        midi_path="/tmp/test.mid",
        bpm=120.0,
        instrument_type="tonal",
        total_notes=1,
        bars_analyzed="1-8",
        notes=[note],
    )
    assert digest.total_notes == 1
    assert digest.notes[0].note == "C4"
