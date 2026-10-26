import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from music_re.core.quality import evaluate_quality
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


def test_silent_reference_is_unavailable(quality_workspace, tmp_path):
    ws, audio = quality_workspace
    gt = tmp_path / 'gt'
    gt.mkdir()
    sf.write(gt / 'other.wav', np.zeros(22050), 22050)
    ws.mark_stage_complete(audio, 'separation', {'other': str(audio)})
    report = evaluate_quality(ws, audio, ground_truth_stems=gt)
    assert not report.is_objective
    assert report.separation_metrics['comparisons'] == 0
