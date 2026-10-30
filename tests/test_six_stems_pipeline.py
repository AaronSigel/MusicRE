from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from music_re.core.midi_project import GM_PROGRAM_MAP, build_combined_midi_project
from music_re.core.profiles import PipelineSettings, resolve_profile
from music_re.schemas import MusicalTimeline, TranscriptionResult
from music_re.workspace import WorkspaceManager


def test_profile_resolution_4_stems():
    fast = resolve_profile('fast')
    assert fast.six_stems is False
    assert fast.stems_count == 4
    assert fast.model_name == 'htdemucs'
    assert fast.supported_stems == ['vocals', 'drums', 'bass', 'other']

    balanced = resolve_profile('balanced', six_stems=False)
    assert balanced.six_stems is False
    assert balanced.stems_count == 4
    assert balanced.model_name == 'htdemucs'

    quality = resolve_profile('quality', six_stems=False)
    assert quality.six_stems is False
    assert quality.stems_count == 4
    assert quality.model_name == 'htdemucs'


def test_profile_resolution_6_stems_and_aliases():
    balanced_6s = resolve_profile('balanced', six_stems=True)
    assert balanced_6s.six_stems is True
    assert balanced_6s.stems_count == 6
    assert balanced_6s.model_name == 'htdemucs_6s'
    assert balanced_6s.supported_stems == ['vocals', 'drums', 'bass', 'other', 'guitar', 'piano']

    quality_6s = resolve_profile('quality', six_stems=True)
    assert quality_6s.six_stems is True
    assert quality_6s.stems_count == 6
    assert quality_6s.model_name == 'htdemucs_6s'

    dense = resolve_profile('dense')
    assert dense.six_stems is True
    assert dense.stems_count == 6
    assert dense.model_name == 'htdemucs_6s'

    six_stems_alias = resolve_profile('six_stems')
    assert six_stems_alias.six_stems is True
    assert six_stems_alias.stems_count == 6
    assert six_stems_alias.model_name == 'htdemucs_6s'


def test_combined_midi_gm_program_assignment(tmp_path):
    midi_dir = tmp_path / "midis"
    midi_dir.mkdir()

    stems = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    stem_midis = {}

    for stem in stems:
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        is_drum = stem == "drums"
        inst = pretty_midi.Instrument(program=0, is_drum=is_drum, name=stem)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60 if not is_drum else 36, start=0.0, end=0.5))
        pm.instruments.append(inst)
        path = midi_dir / f"{stem}.mid"
        pm.write(str(path))
        stem_midis[stem] = str(path)

    out_midi = midi_dir / "all_stems.mid"
    build_combined_midi_project(stem_midis, out_midi, bpm=120.0)

    combined_pm = pretty_midi.PrettyMIDI(str(out_midi))
    inst_by_name = {inst.name.lower(): inst for inst in combined_pm.instruments}

    assert inst_by_name["guitar"].program == 25
    assert inst_by_name["guitar"].is_drum is False

    assert inst_by_name["piano"].program == 0
    assert inst_by_name["piano"].is_drum is False

    assert inst_by_name["bass"].program == 33
    assert inst_by_name["bass"].is_drum is False

    assert inst_by_name["vocals"].program == 54
    assert inst_by_name["vocals"].is_drum is False

    assert inst_by_name["other"].program == 81
    assert inst_by_name["other"].is_drum is False

    assert inst_by_name["drums"].is_drum is True


def test_pipeline_6_stem_transcription_routing(tmp_path, monkeypatch):
    from music_re.core.pipeline import run_pipeline
    from music_re.schemas import StemSeparationResult

    source = tmp_path / "test_track.wav"
    sf.write(str(source), np.zeros((44100, 2), dtype=np.float32), 44100)

    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_dir = ws.get_track_dir(source)
    stems_dir = track_dir / "stems"
    stems_dir.mkdir(parents=True)

    six_stem_names = ["vocals", "drums", "bass", "other", "guitar", "piano"]
    stem_paths = {}
    for name in six_stem_names:
        p = stems_dir / f"{name}.wav"
        sf.write(str(p), np.zeros((22050, 2), dtype=np.float32), 22050)
        stem_paths[name] = str(p)

    def fake_separate(*args, **kwargs):
        return StemSeparationResult(
            workspace_dir=str(track_dir),
            model_used="htdemucs_6s",
            stems=stem_paths,
        )

    monkeypatch.setattr("music_re.core.pipeline.separate_stems", fake_separate)

    # Mock structure analysis to keep test fast and offline
    fake_structure = MagicMock()
    fake_structure.analyze.return_value = MusicalTimeline(
        duration_sec=1.0, bpm=120.0, beats=[0.0, 0.5], downbeats=[0.0],
        sections=[], backend="librosa"
    )
    monkeypatch.setattr("music_re.core.pipeline.get_structure_analyzer", lambda *a, **k: fake_structure)

    routed_backends = {}

    def fake_get_transcriber_backend(mode="tonal", name="auto", **kwargs):
        mock_transcriber = MagicMock()

        def fake_transcribe(audio_path, output_midi_path, **tr_kwargs):
            stem_name = tr_kwargs.get("stem_name")
            routed_backends[stem_name] = (mode, name)

            # Write a dummy midi so downstream file validity checks pass
            pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            inst = pretty_midi.Instrument(program=0, is_drum=(stem_name == "drums"))
            inst.notes.append(pretty_midi.Note(velocity=70, pitch=60, start=0.0, end=0.25))
            pm.instruments.append(inst)
            out_p = Path(output_midi_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            pm.write(str(out_p))

            raw_path = str(out_p).replace(".mid", "_raw.mid")
            pm.write(raw_path)

            return TranscriptionResult(
                status="notes",
                midi_path=str(out_p),
                raw_midi_path=raw_path,
                note_count=1,
                backend=name or mode,
            )

        mock_transcriber.transcribe.side_effect = fake_transcribe
        return mock_transcriber

    monkeypatch.setattr("music_re.core.pipeline.get_transcriber_backend", fake_get_transcriber_backend)

    res = run_pipeline(source, workspace=ws, profile="dense")
    assert ws.get_manifest(source).status == "completed"
    assert "guitar" in res["stems"]
    assert "piano" in res["stems"]
    assert "all_stems" in res["midis"]

    # Verify routing:
    # vocals -> crepe_vocal
    assert routed_backends["vocals"] == ("tonal", "crepe_vocal")
    # drums -> adtof
    assert routed_backends["drums"] == ("drums", "adtof")
    # bass -> basic_pitch
    assert routed_backends["bass"] == ("tonal", "basic_pitch")
    # guitar -> basic_pitch
    assert routed_backends["guitar"] == ("tonal", "basic_pitch")
    # piano -> bytedance_piano
    assert routed_backends["piano"] == ("tonal", "bytedance_piano")
    # other -> basic_pitch
    assert routed_backends["other"] == ("tonal", "basic_pitch")


def test_cli_help_recommends_six_stems():
    from typer.testing import CliRunner
    from music_re.cli import app

    runner = CliRunner()
    res_split = runner.invoke(app, ["split", "--help"])
    assert res_split.exit_code == 0
    assert "--six-stems" in res_split.output
    assert "guitar and piano" in res_split.output

    res_deconstruct = runner.invoke(app, ["deconstruct", "--help"])
    assert res_deconstruct.exit_code == 0
    assert "--six-stems" in res_deconstruct.output
    assert "guitar and piano" in res_deconstruct.output

