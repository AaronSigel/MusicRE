"""DAWproject export module for MusicRE.

Generates .dawproject exchange files containing audio stems, MIDI tracks,
markers, tempo, and time signature for direct import into supported DAWs.
"""

from pathlib import Path
from typing import Optional, Union

import pretty_midi
import soundfile as sf
from dawproject import (
    Application,
    Arrangement,
    Audio,
    Clip,
    Clips,
    ContentType,
    DawProject,
    FileReference,
    Lanes,
    Marker,
    Markers,
    MetaData,
    Note,
    Notes,
    Project,
    RealParameter,
    Referenceable,
    TimeSignatureParameter,
    Track,
    Transport,
    Unit,
)

from music_re.schemas import MusicalTimeline
from music_re.workspace import WorkspaceManager


class EmbeddedFiles(dict):
    """Container for embedded files that avoids key collisions on identical file bytes."""

    def __init__(self, entries=None):
        super().__init__()
        self._entries = list(entries or [])

    def append(self, data: bytes, path: str) -> None:
        self._entries.append((data, path))

    def items(self):
        return iter(self._entries)


def export_dawproject(
    workspace: WorkspaceManager,
    file_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """Export workspace deliverables to a .dawproject exchange file."""
    Referenceable.reset_id()

    manifest = workspace.get_manifest(file_path)
    if not manifest:
        raise ValueError(f"No manifest found for {file_path}")

    timeline: Optional[MusicalTimeline] = None
    if hasattr(workspace, "load_timeline"):
        timeline = workspace.load_timeline(file_path)
    if timeline is None and getattr(manifest, "timeline", None):
        raw_tl = manifest.timeline
        if isinstance(raw_tl, MusicalTimeline):
            timeline = raw_tl
        elif isinstance(raw_tl, dict):
            timeline = MusicalTimeline.model_validate(raw_tl)

    track_duration = float(timeline.duration_sec) if timeline and timeline.duration_sec else 0.0

    structure_tracks = []
    arrangement_lanes = []
    embedded_files = EmbeddedFiles()

    # 1. Audio stem tracks and clips
    stems = getattr(manifest, "stems", {}) or {}
    for role, stem_file in stems.items():
        stem_path = Path(stem_file)
        if not stem_path.is_file():
            continue

        info = sf.info(stem_path)
        dur = float(info.duration)
        track_duration = max(track_duration, dur)

        audio_track = Track(name=str(role), content_type=[ContentType.AUDIO])
        rel_audio_path = f"audio/{role}.wav"

        audio_content = Audio(
            sample_rate=info.samplerate,
            channels=info.channels,
            duration=dur,
            file=FileReference(path=rel_audio_path, external=False),
        )
        audio_clip = Clip(time=0.0, duration=dur, content=audio_content)
        audio_clips = Clips(track=audio_track, clips=[audio_clip])

        structure_tracks.append(audio_track)
        arrangement_lanes.append(audio_clips)
        embedded_files.append(stem_path.read_bytes(), rel_audio_path)

    # 2. MIDI tracks and clips
    midis = getattr(manifest, "midis", {}) or {}
    for role, midi_file in midis.items():
        midi_path = Path(midi_file)
        if not midi_path.is_file():
            continue

        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception:
            continue

        notes = []
        for inst in pm.instruments:
            for n in inst.notes:
                notes.append(
                    Note(
                        time=float(n.start),
                        duration=float(max(0.0, n.end - n.start)),
                        key=int(n.pitch),
                        vel=float(n.velocity),
                    )
                )
        notes.sort(key=lambda item: item.time)
        notes_content = Notes(notes=notes)

        if notes:
            last_end = max(n.time + n.duration for n in notes)
            clip_dur = max(last_end, track_duration)
        else:
            clip_dur = track_duration

        track_name = f"{role} (MIDI)" if role in stems else str(role)
        midi_track = Track(name=track_name, content_type=[ContentType.NOTES])
        midi_clip = Clip(time=0.0, duration=clip_dur, content=notes_content)
        midi_clips = Clips(track=midi_track, clips=[midi_clip])

        structure_tracks.append(midi_track)
        arrangement_lanes.append(midi_clips)

    # 3. Markers from timeline sections and chords
    marker_list = []
    if timeline and getattr(timeline, "sections", None):
        for s in timeline.sections:
            marker_list.append(Marker(time=float(s.start_sec), name=str(s.label)))

    if timeline and getattr(timeline, "chords", None):
        prev_chord = None
        for c in timeline.chords:
            c_chord = getattr(c, "chord", None) or (c.get("chord") if isinstance(c, dict) else None)
            c_time = getattr(c, "start_sec", None) if hasattr(c, "start_sec") else (c.get("start_sec") if isinstance(c, dict) else 0.0)
            if c_chord and c_chord != "N/A" and c_chord != prev_chord:
                marker_list.append(Marker(time=float(c_time), name=str(c_chord)))
                prev_chord = c_chord

    markers = Markers(markers=marker_list) if marker_list else None

    # 4. Transport tempo & time signature
    tempo_bpm = 120.0
    num = 4
    den = 4
    if timeline is not None:
        if getattr(timeline, "bpm", None) and timeline.bpm > 0:
            tempo_bpm = float(timeline.bpm)
        if (
            getattr(timeline, "downbeats", None)
            and getattr(timeline, "beats", None)
            and len(timeline.downbeats) >= 2
            and len(timeline.beats) >= 2
        ):
            first_bar = [
                b for b in timeline.beats
                if timeline.downbeats[0] <= b < timeline.downbeats[1]
            ]
            if len(first_bar) in (2, 3, 4, 5, 6, 7, 9, 12):
                num = len(first_bar)

    transport = Transport(
        tempo=RealParameter(value=tempo_bpm, unit=Unit.BPM),
        time_signature=TimeSignatureParameter(numerator=num, denominator=den),
    )

    # 5. Arrangement & Project
    arrangement = Arrangement(
        lanes=Lanes(lanes=arrangement_lanes),
        markers=markers,
    )

    project = Project(
        application=Application(name="MusicRE", version="0.1.0"),
        transport=transport,
        structure=structure_tracks,
        arrangement=arrangement,
    )

    # 6. Metadata
    meta_title = Path(file_path).stem
    meta_artist = None
    if hasattr(workspace, "load_metadata"):
        ws_meta = workspace.load_metadata(file_path)
        if ws_meta and isinstance(ws_meta, dict):
            meta_title = ws_meta.get("title") or meta_title
            meta_artist = ws_meta.get("artist")

    metadata = MetaData(title=meta_title, artist=meta_artist)

    # 7. Output path resolution and save
    out_p = Path(output_path).resolve()
    if out_p.is_dir():
        out_p = out_p / f"{Path(file_path).stem}.dawproject"
    elif out_p.suffix != ".dawproject":
        out_p = out_p.with_suffix(".dawproject")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    DawProject.save(project, metadata, embedded_files, str(out_p))
    return out_p
