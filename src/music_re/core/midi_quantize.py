from typing import Dict, List, Optional
import numpy as np
import pretty_midi

from music_re.schemas import MusicalTimeline


def build_quantize_grid(
    timeline: Optional[MusicalTimeline] = None,
    bpm: float = 120.0,
    grid_subdivisions: int = 4,
    max_duration: float = 600.0,
) -> np.ndarray:
    """Build fine-grained timestamps for quantization using dynamic beats or static metronome."""
    if timeline and timeline.beats and len(timeline.beats) >= 2:
        beats = np.array(sorted(timeline.beats), dtype=np.float64)
        sub_grid = []

        # Subdivide each beat interval
        for i in range(len(beats) - 1):
            b0 = beats[i]
            b1 = beats[i + 1]
            sub_points = np.linspace(b0, b1, grid_subdivisions, endpoint=False)
            sub_grid.extend(sub_points)
        sub_grid.append(beats[-1])

        # Extend beyond last beat if track is longer
        avg_step = float(np.mean(np.diff(beats))) / grid_subdivisions
        if avg_step <= 0.0:
            avg_step = (60.0 / bpm) / grid_subdivisions

        last_t = sub_grid[-1]
        while last_t < max_duration + 5.0:
            last_t += avg_step
            sub_grid.append(last_t)

        return np.array(sub_grid, dtype=np.float64)

    # Static metronome fallback
    effective_bpm = bpm if bpm > 0 else 120.0
    step = (60.0 / effective_bpm) / grid_subdivisions
    return np.arange(0.0, max_duration + 5.0, step, dtype=np.float64)


def snap_to_nearest_grid(time_val: float, grid: np.ndarray) -> float:
    """Find the closest timestamp in a sorted grid."""
    idx = np.searchsorted(grid, time_val)
    if idx == 0:
        return float(grid[0])
    if idx >= len(grid):
        return float(grid[-1])
    before = grid[idx - 1]
    after = grid[idx]
    if (time_val - before) <= (after - time_val):
        return float(before)
    return float(after)


def quantize_to_timeline(
    pm: pretty_midi.PrettyMIDI,
    timeline: Optional[MusicalTimeline] = None,
    bpm: float = 120.0,
    grid_subdivisions: int = 4,
    min_duration: float = 0.05,
) -> pretty_midi.PrettyMIDI:
    """Snap note onsets to dynamic timeline subdivisions and resolve same-pitch polyphonic overlaps."""
    max_t = 0.0
    for inst in pm.instruments:
        for n in inst.notes:
            if n.end > max_t:
                max_t = n.end
    if timeline and timeline.duration_sec > max_t:
        max_t = timeline.duration_sec

    grid = build_quantize_grid(
        timeline=timeline,
        bpm=bpm,
        grid_subdivisions=grid_subdivisions,
        max_duration=max_t,
    )

    limit_duration = timeline.duration_sec if timeline and timeline.duration_sec > 0 else 0.0

    for inst in pm.instruments:
        valid_notes: List[pretty_midi.Note] = []
        for n in inst.notes:
            if n.end - n.start < min_duration:
                continue

            # Discard notes that start beyond track boundary
            if limit_duration > 0 and n.start >= limit_duration:
                continue

            snapped_start = snap_to_nearest_grid(n.start, grid)
            if limit_duration > 0 and snapped_start + min_duration > limit_duration:
                snapped_start = max(0.0, limit_duration - min_duration)
                snapped_end = limit_duration
            else:
                snapped_end = max(snapped_start + min_duration, n.end)
                if limit_duration > 0 and snapped_end > limit_duration:
                    snapped_end = limit_duration

            if snapped_end <= snapped_start:
                continue

            valid_notes.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=snapped_start,
                    end=snapped_end,
                )
            )

        # Eliminate overlapping notes of identical pitch
        pitch_groups: Dict[int, List[pretty_midi.Note]] = {}
        for n in valid_notes:
            pitch_groups.setdefault(n.pitch, []).append(n)

        cleaned_notes: List[pretty_midi.Note] = []
        for pitch, p_notes in pitch_groups.items():
            p_notes.sort(key=lambda x: (x.start, x.end))
            resolved: List[pretty_midi.Note] = []

            for n in p_notes:
                if not resolved:
                    resolved.append(n)
                    continue

                prev = resolved[-1]
                if n.start < prev.end:
                    # Overlap or micro-gap
                    if abs(n.start - prev.start) < min_duration:
                        prev.end = max(prev.end, n.end)
                        prev.velocity = max(prev.velocity, n.velocity)
                    else:
                        prev.end = n.start
                        if prev.end > prev.start:
                            resolved.append(n)
                        else:
                            resolved[-1] = n
                else:
                    resolved.append(n)

            cleaned_notes.extend(resolved)

        cleaned_notes.sort(key=lambda n: (n.start, n.pitch))
        inst.notes = cleaned_notes

    return pm
