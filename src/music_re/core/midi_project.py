import bisect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pretty_midi

from music_re.schemas import MusicalTimeline

logger = logging.getLogger(__name__)

GM_PROGRAM_MAP = {
    "bass": 33,      # Electric Bass (finger)
    "piano": 0,      # Acoustic Grand Piano
    "guitar": 25,    # Acoustic Guitar (steel)
    "vocals": 54,    # Synth Voice
    "other": 81,     # Lead 2 (sawtooth)
    "drums": 0,      # Channel 10 Standard Kit
}


def build_combined_midi_project(
    stem_midis: Dict[str, Union[Path, str]],
    output_path: Union[Path, str],
    timeline: Optional[MusicalTimeline] = None,
    bpm: float = 120.0,
) -> Path:
    """Combine individual stem MIDI files into a single multi-track Standard MIDI File (SMF Type 1)."""
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    effective_bpm = timeline.bpm if timeline and timeline.bpm > 0 else bpm
    combined_pm = pretty_midi.PrettyMIDI(initial_tempo=effective_bpm)

    # Set up dynamic tempo map from timeline beats if available
    if timeline and len(timeline.beats) >= 2:
        unique_beats = sorted({float(t) for t in timeline.beats if t >= 0})
        if len(unique_beats) >= 2:
            tick_scales = []
            current_bpm = effective_bpm
            current_tick = 0
            
            # Initial beat tempo
            first_interval = unique_beats[1] - unique_beats[0]
            if first_interval > 0:
                current_bpm = 60.0 / first_interval
            tick_scale = 60.0 / (current_bpm * combined_pm.resolution)
            tick_scales.append((0, tick_scale))
            
            # For subsequent beats, add tempo changes at exact ticks
            for i in range(1, len(unique_beats) - 1):
                b_prev = unique_beats[i - 1]
                b_curr = unique_beats[i]
                b_next = unique_beats[i + 1]
                interval = b_next - b_curr
                
                # Advance tick count by previous beat duration
                current_tick += int(round(combined_pm.resolution))
                
                if interval > 0:
                    inst_bpm = 60.0 / interval
                    if abs(inst_bpm - current_bpm) > 0.5:
                        current_bpm = inst_bpm
                        tick_scale = 60.0 / (current_bpm * combined_pm.resolution)
                        tick_scales.append((current_tick, tick_scale))

            combined_pm._tick_scales = tick_scales
            max_tick = current_tick + int(round(combined_pm.resolution * 100))
            combined_pm._update_tick_to_time(max_tick)

    # Time signature from timeline downbeats if available
    if timeline and len(timeline.downbeats) >= 2 and len(timeline.beats) >= 2:
        # Estimate numerator as average beats per downbeat
        beats_in_bar = round((timeline.downbeats[1] - timeline.downbeats[0]) / max(1e-4, np.mean(np.diff(timeline.beats))))
        num = max(1, min(16, int(beats_in_bar)))
        combined_pm.time_signature_changes = [pretty_midi.TimeSignature(numerator=num, denominator=4, time=0.0)]
    else:
        combined_pm.time_signature_changes = [pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)]

    for stem_name, stem_path in stem_midis.items():
        if stem_name.lower() == "all_stems":
            continue
        stem_p = Path(stem_path).resolve()
        if not stem_p.is_file():
            raise ValueError(f"Missing stem MIDI file for '{stem_name}': {stem_p}")

        try:
            stem_pm = pretty_midi.PrettyMIDI(str(stem_p), initial_tempo=effective_bpm)
        except Exception as e:
            raise ValueError(f"Corrupt or unreadable MIDI file for '{stem_name}' at {stem_p}: {e}") from e

        s_key = stem_name.lower()
        is_drum = s_key in ("drums", "percussion")
        program = GM_PROGRAM_MAP.get(s_key, 0)

        # Merge all notes, pitch bends, control changes from this stem's instruments into one designated track
        stem_instrument = pretty_midi.Instrument(
            program=program,
            is_drum=is_drum,
            name=stem_name.capitalize(),
        )

        for src_inst in stem_pm.instruments:
            for note in src_inst.notes:
                stem_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end,
                    )
                )
            for pb in src_inst.pitch_bends:
                stem_instrument.pitch_bends.append(
                    pretty_midi.PitchBend(pitch=pb.pitch, time=pb.time)
                )
            for cc in src_inst.control_changes:
                stem_instrument.control_changes.append(
                    pretty_midi.ControlChange(number=cc.number, value=cc.value, time=cc.time)
                )

        stem_instrument.notes.sort(key=lambda n: (n.start, n.pitch))
        stem_instrument.pitch_bends.sort(key=lambda pb: pb.time)
        stem_instrument.control_changes.sort(key=lambda cc: cc.time)
        combined_pm.instruments.append(stem_instrument)

    # Post-process non-drum instruments: normalize pitch bends and resolve harmonic clashes
    for inst in combined_pm.instruments:
        if not inst.is_drum:
            normalize_instrument_pitch_bends(inst)

    resolve_harmonic_dissonance(combined_pm.instruments)

    for inst in combined_pm.instruments:
        if not inst.is_drum:
            deduplicate_overlapping_notes(inst)
            normalize_instrument_pitch_bends(inst)

    combined_pm.write(str(out_p))
    return out_p


def normalize_instrument_pitch_bends(instrument: pretty_midi.Instrument) -> None:
    """Ensure pitch bend resets to 0 at note endings and strip static tuning detune."""
    if not instrument.pitch_bends:
        return

    # Check for static detune (multiple pitch bends all stuck at identical non-zero value, e.g. constant 1365 = +33 cents)
    pb_vals = {pb.pitch for pb in instrument.pitch_bends}
    if len(instrument.pitch_bends) >= 2 and len(pb_vals) == 1 and 0 not in pb_vals:
        logger.info("Stripping static pitch bend (%d) on %s", next(iter(pb_vals)), instrument.name)
        instrument.pitch_bends = []
        return

    if not instrument.notes:
        return

    instrument.pitch_bends.sort(key=lambda pb: pb.time)
    pb_times = [pb.time for pb in instrument.pitch_bends]
    existing_times = set(pb_times)
    new_resets = []

    for note in instrument.notes:
        idx = bisect.bisect_right(pb_times, note.end) - 1
        if idx >= 0:
            last_pb = instrument.pitch_bends[idx]
            if last_pb.pitch != 0:
                t = round(note.end, 6)
                if t not in existing_times:
                    new_resets.append(pretty_midi.PitchBend(pitch=0, time=t))
                    existing_times.add(t)

    if new_resets:
        instrument.pitch_bends.extend(new_resets)
        instrument.pitch_bends.sort(key=lambda pb: pb.time)


def resolve_harmonic_dissonance(instruments: List[pretty_midi.Instrument]) -> None:
    """Resolve minor 2nd (1-semitone) clashes between simultaneous notes across non-drum stems."""
    non_drum_insts = [inst for inst in instruments if not inst.is_drum]
    if len(non_drum_insts) < 2:
        return

    STEM_PRIORITY = {
        "vocals": 5,
        "bass": 4,
        "piano": 3,
        "guitar": 2,
        "other": 1,
    }

    max_passes = 4
    for _ in range(max_passes):
        clash_resolved = False

        for i in range(len(non_drum_insts)):
            for j in range(i + 1, len(non_drum_insts)):
                inst_a = non_drum_insts[i]
                inst_b = non_drum_insts[j]
                prio_a = STEM_PRIORITY.get(inst_a.name.lower(), 2)
                prio_b = STEM_PRIORITY.get(inst_b.name.lower(), 2)

                remove_a = set()
                remove_b = set()

                for n_a in inst_a.notes:
                    if id(n_a) in remove_a:
                        continue
                    for n_b in inst_b.notes:
                        if id(n_b) in remove_b:
                            continue

                        # Check time overlap
                        overlap_start = max(n_a.start, n_b.start)
                        overlap_end = min(n_a.end, n_b.end)
                        if (overlap_end - overlap_start) <= 0.03:
                            continue

                        # Check 1-semitone clash (minor 2nd)
                        if abs(n_a.pitch - n_b.pitch) != 1:
                            continue

                        # Compare strengths
                        dur_a = n_a.end - n_a.start
                        dur_b = n_b.end - n_b.start
                        score_a = (n_a.velocity / 127.0) * min(dur_a, 2.0) + (prio_a * 0.05)
                        score_b = (n_b.velocity / 127.0) * min(dur_b, 2.0) + (prio_b * 0.05)

                        if score_a >= score_b:
                            dom_n, clash_n = n_a, n_b
                            dom_inst, clash_inst = inst_a, inst_b
                            is_b_clash = True
                        else:
                            dom_n, clash_n = n_b, n_a
                            dom_inst, clash_inst = inst_b, inst_a
                            is_b_clash = False

                        clash_dur = clash_n.end - clash_n.start

                        # Filter ghost overtones: lower velocity (< 55), short (< 0.20s), or significantly weaker
                        if (
                            clash_n.velocity < 55
                            or clash_dur < 0.20
                            or clash_n.velocity <= 0.75 * dom_n.velocity
                            or clash_dur <= 0.60 * (dom_n.end - dom_n.start)
                        ):
                            logger.info(
                                "Harmonic dissonance: filtered ghost overtone pitch %d (vel=%d, dur=%.2fs) on %s clashing with pitch %d on %s",
                                clash_n.pitch, clash_n.velocity, clash_dur, clash_inst.name,
                                dom_n.pitch, dom_inst.name,
                            )
                            if is_b_clash:
                                remove_b.add(id(clash_n))
                            else:
                                remove_a.add(id(clash_n))
                        else:
                            # Nudge clashing note pitch to unison with dominant note to avoid dissonance
                            old_pitch = clash_n.pitch
                            clash_n.pitch = dom_n.pitch
                            logger.info(
                                "Harmonic dissonance: nudged clashing note pitch %d -> %d on %s to resolve clash with %s",
                                old_pitch, clash_n.pitch, clash_inst.name, dom_inst.name,
                            )

                        clash_resolved = True

                if remove_a:
                    inst_a.notes = [n for n in inst_a.notes if id(n) not in remove_a]
                if remove_b:
                    inst_b.notes = [n for n in inst_b.notes if id(n) not in remove_b]

        if not clash_resolved:
            break


def deduplicate_overlapping_notes(inst: pretty_midi.Instrument) -> None:
    """Merge overlapping notes with identical pitch on the same instrument."""
    if not inst.notes:
        return
    inst.notes.sort(key=lambda n: (n.pitch, n.start))
    merged = []
    for note in inst.notes:
        if merged and merged[-1].pitch == note.pitch and note.start < merged[-1].end:
            merged[-1].end = max(merged[-1].end, note.end)
            merged[-1].velocity = max(merged[-1].velocity, note.velocity)
        else:
            merged.append(note)
    merged.sort(key=lambda n: (n.start, n.pitch))
    inst.notes = merged
