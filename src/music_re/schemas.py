from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkspaceManifest(BaseModel):
    track_id: str
    input_file: str
    status: str = Field(default="pending", description="'pending' | 'processing' | 'completed' | 'failed'")
    current_stage: str = Field(default="none")
    completed_stages: List[str] = Field(default_factory=list)
    stage_errors: Dict[str, str] = Field(default_factory=dict)
    stems: Dict[str, str] = Field(default_factory=dict)
    analysis: Optional[Dict[str, Any]] = None
    timeline: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    midis: Dict[str, str] = Field(default_factory=dict)


class QualityReport(BaseModel):
    is_objective: bool = Field(description="True if calculated against ground truth, False if diagnostic")
    separation_metrics: Dict[str, Any] = Field(default_factory=dict)
    transcription_metrics: Dict[str, Any] = Field(default_factory=dict)
    audio_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    evaluated_at: str = Field(description="ISO timestamp")


class TimelineSection(BaseModel):
    label: str = Field(description="Section label, e.g. 'intro', 'verse', 'chorus', 'bridge', 'outro'")
    start_sec: float = Field(description="Start time in seconds")
    end_sec: float = Field(description="End time in seconds")
    confidence: float = Field(default=1.0, description="Detection confidence 0.0 - 1.0")


class MusicalTimeline(BaseModel):
    duration_sec: float
    bpm: float
    beats: List[float] = Field(description="Beat timestamps in seconds")
    downbeats: List[float] = Field(description="Downbeat / bar start timestamps in seconds")
    sections: List[TimelineSection] = Field(default_factory=list)


class TrackAnalysis(BaseModel):
    file_path: str
    duration_sec: float
    bpm: float = Field(description="Tempo in beats per minute")
    key: Optional[str] = Field(default="N/A", description="Musical key, e.g. 'F# minor'")
    key_confidence: float
    lufs: float = Field(description="Integrated loudness of track")
    beat_times: List[float] = Field(description="Beat timestamps in seconds")


class StemSeparationResult(BaseModel):
    workspace_dir: str
    model_used: str
    stems: Dict[str, str] = Field(description="Mapping of stem_name to absolute wav path")
    quality_note: Optional[str] = Field(default=None, description="Quality or bleed notice for model/mode")


class NoteEvent(BaseModel):
    note: str          # e.g. "C#3"
    pitch: int         # MIDI note number, e.g. 49
    bar: int           # Bar index (1-indexed)
    beat_in_bar: float # Beat position within bar (1.0 = on the one)
    duration_beats: float
    velocity: int


class MidiDigest(BaseModel):
    midi_path: str
    bpm: float
    instrument_type: str  # "tonal" | "drums"
    total_notes: int
    bars_analyzed: str    # e.g. "1-8"
    notes: List[NoteEvent]
