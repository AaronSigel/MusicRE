import json
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from music_re.core.quality import evaluate_quality
from music_re.core.quality_midi import transcription_quality
from music_re.workspace import WorkspaceManager


@pytest.fixture
def quality_workspace(tmp_path):
    audio = tmp_path / 'mix.wav'
    sf.write(audio, np.zeros(22050), 22050)
    return WorkspaceManager(tmp_path / 'cache'), audio


def write_note(path, pitch=60):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, pitch, .1, .6))
    pm.instruments.append(inst)
    pm.write(str(path))


def write_drum_note(path, pitch, start, append_to=None):
    source = append_to if append_to is not None else path
    if append_to is not None and Path(source).exists():
        pm = pretty_midi.PrettyMIDI(str(source))
    else:
        pm = pretty_midi.PrettyMIDI()
        pm.instruments.append(pretty_midi.Instrument(0, is_drum=True))
    pm.instruments[0].notes.append(pretty_midi.Note(100, pitch, start, start + .3))
    pm.write(str(path))


def test_empty_reference_directory_is_not_objective(quality_workspace, tmp_path):
    ws, audio = quality_workspace
    gt = tmp_path / 'gt'
    gt.mkdir()
    report = evaluate_quality(ws, audio, ground_truth_stems=gt)
    assert not report.is_objective
    assert report.separation_metrics['comparisons'] == 0
    assert report.separation_metrics['warnings']


def test_wrong_pitch_is_not_a_correct_transcription(quality_workspace, tmp_path):
    ws, audio = quality_workspace
    reference, estimate = tmp_path / 'ref.mid', tmp_path / 'est.mid'
    write_note(reference, 60)
    write_note(estimate, 61)
    ws.mark_stage_complete(audio, 'transcription', {'vocals': str(estimate)})
    report = evaluate_quality(ws, audio, ground_truth_midi=reference)
    assert report.is_objective
    assert report.transcription_metrics['pitched']['onset_pitch_f1'] == 0


def test_combined_midi_is_not_double_counted(quality_workspace, tmp_path):
    ws, audio = quality_workspace
    estimate = tmp_path / 'renamed.mid'
    write_note(estimate)
    ws.mark_stage_complete(audio, 'transcription', {'vocals': str(estimate), 'all_stems': str(estimate)})
    report = evaluate_quality(ws, audio)
    assert report.transcription_metrics['total_notes'] == 1


def test_corrupt_estimate_cannot_claim_objective(quality_workspace, tmp_path):
    ws, audio = quality_workspace
    reference, estimate = tmp_path / 'ref.mid', tmp_path / 'bad.mid'
    write_note(reference)
    estimate.write_bytes(b'bad')
    ws.mark_stage_complete(audio, 'transcription', {'vocals': str(estimate)})
    report = evaluate_quality(ws, audio, ground_truth_midi=reference)
    assert not report.is_objective
    assert report.transcription_metrics['warnings']


def test_silence_report_has_strict_json_and_no_fake_snr(quality_workspace):
    ws, audio = quality_workspace
    ws.mark_stage_complete(audio, 'separation', {'other': str(audio)})
    report = evaluate_quality(ws, audio)
    assert 'snr_estimate_db' not in report.audio_diagnostics
    assert report.audio_diagnostics['rms_to_p10_amplitude_db'] is None
    json.dumps(report.model_dump(), allow_nan=False)


def test_drum_class_breakdown_scores_kick_snare_separately(tmp_path):
    reference, estimate = tmp_path / 'ref.mid', tmp_path / 'est.mid'
    write_drum_note(reference, pitch=36, start=0.0)   # kick
    write_drum_note(reference, pitch=38, start=1.0, append_to=reference)  # snare, missed by estimate
    write_drum_note(estimate, pitch=36, start=0.0)     # matches the kick
    write_drum_note(estimate, pitch=42, start=2.0, append_to=estimate)    # false-positive hihat

    metrics = transcription_quality({'drums': str(estimate)}, ground_truth=str(reference))

    assert metrics['drums_by_class']['kick']['f1'] == 1.0
    assert metrics['drums_by_class']['snare']['f1'] == 0.0
    assert metrics['drums_by_class']['hihat']['f1'] == 0.0
    # A 0.0 f1 is ambiguous without knowing how many events existed on each
    # side: "ref had notes, none matched" vs "ref had no notes at all" read
    # very differently. Expose the raw counts, not just the derived score.
    assert metrics['drums_by_class']['snare'] == {
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'ref_count': 1, 'est_count': 0,
    }
    assert metrics['drums_by_class']['hihat'] == {
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'ref_count': 0, 'est_count': 1,
    }


def test_drum_class_breakdown_counts_other_gm_kit_pitches_in_each_class(tmp_path):
    # Review I4: real Slakh drum MIDI uses the full GM kit (e.g. kick 35,
    # not just 36). A note at a GM variant pitch must still be scored
    # within its class, not silently excluded from every class because it
    # fails an exact match against only {36, 38, 42}. If pitch 35 isn't
    # recognized as a kick at all, 'kick' is absent from drums_by_class
    # here (both sides are empty at pitch 36 exactly) -- that absence is
    # exactly what would wrongly hide this reference note from scoring.
    reference, estimate = tmp_path / 'ref2.mid', tmp_path / 'est2.mid'
    write_drum_note(reference, pitch=35, start=0.0)  # GM "Acoustic Bass Drum", same class as kick
    write_drum_note(estimate, pitch=35, start=0.0)    # matches exactly

    metrics = transcription_quality({'drums': str(estimate)}, ground_truth=str(reference))

    assert metrics['drums_by_class']['kick']['f1'] == 1.0


def test_drum_class_f1_scores_gm_aliases_correctly(tmp_path):
    reference, estimate = tmp_path / 'ref_aliases.mid', tmp_path / 'est_aliases.mid'
    write_drum_note(reference, pitch=35, start=0.0)
    write_drum_note(reference, pitch=37, start=1.0, append_to=reference)
    write_drum_note(reference, pitch=44, start=2.0, append_to=reference)

    write_drum_note(estimate, pitch=36, start=0.0)
    write_drum_note(estimate, pitch=38, start=1.0, append_to=estimate)
    write_drum_note(estimate, pitch=42, start=2.0, append_to=estimate)

    metrics = transcription_quality({'drums': str(estimate)}, ground_truth=str(reference))

    assert metrics['drums_by_class']['kick']['f1'] == 1.0
    assert metrics['drums_by_class']['snare']['f1'] == 1.0
    assert metrics['drums_by_class']['hihat']['f1'] == 1.0


def test_silent_reference_is_unavailable(quality_workspace, tmp_path):
    ws, audio = quality_workspace
    gt = tmp_path / 'gt'
    gt.mkdir()
    sf.write(gt / 'other.wav', np.zeros(22050), 22050)
    ws.mark_stage_complete(audio, 'separation', {'other': str(audio)})
    report = evaluate_quality(ws, audio, ground_truth_stems=gt)
    assert not report.is_objective
    assert report.separation_metrics['comparisons'] == 0
