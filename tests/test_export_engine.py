from pathlib import Path
import zipfile
import pretty_midi
import soundfile as sf
import numpy as np
import pytest

from music_re.core.export import export_bundle
from music_re.schemas import MusicalTimeline, QualityReport, TimelineSection
from music_re.workspace import WorkspaceManager

@pytest.fixture
def populated_workspace(tmp_path: Path, complete_workspace):
    # Use a fully registered fixture; files alone do not constitute completion.
    return complete_workspace

def test_export_bundle_directory(tmp_path: Path, populated_workspace):
    ws, track_wav = populated_workspace
    export_out = tmp_path / "export_output"

    res = export_bundle(ws, track_wav, export_out, archive=False)
    assert res.is_dir()
    assert (res / "timeline.json").is_file()
    assert (res / "manifest.json").is_file()
    assert (res / "quality_report.json").is_file()
    assert (res / "stems" / "vocals.wav").is_file()
    assert (res / "midis" / "all_stems.mid").is_file()

def test_export_bundle_zip_archive(tmp_path: Path, populated_workspace):
    ws, track_wav = populated_workspace
    export_zip_target = tmp_path / "export_archive.zip"

    res = export_bundle(ws, track_wav, export_zip_target, archive=True)
    assert res.is_file()
    assert res.suffix == ".zip"

    with zipfile.ZipFile(str(res), "r") as zf:
        names = zf.namelist()
        assert any("timeline.json" in n for n in names)
        assert any("manifest.json" in n for n in names)
        assert any("quality_report.json" in n for n in names)
        assert any("all_stems.mid" in n for n in names)


def test_empty_workspace_cannot_export(tmp_path):
    ws = WorkspaceManager(tmp_path / 'cache')
    audio = tmp_path / 'song.wav'
    sf.write(audio, np.zeros(22050), 22050)
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='Incomplete'):
        export_bundle(ws, audio, out)
    assert not out.exists()


@pytest.mark.parametrize('artifact', ['stem', 'midi', 'timeline', 'quality'])
def test_missing_required_artifact_fails_before_export(populated_workspace, tmp_path, artifact):
    ws, audio = populated_workspace
    m = ws.get_manifest(audio)
    paths = {'stem': m.stems['vocals'], 'midi': m.midis['all_stems'],
             'timeline': ws.get_timeline_path(audio), 'quality': ws.get_quality_report_path(audio)}
    Path(paths[artifact]).unlink()
    out = tmp_path / 'incomplete'
    with pytest.raises(ValueError, match='Incomplete'):
        export_bundle(ws, audio, out)
    assert not out.exists()


def test_bundle_manifest_is_portable(populated_workspace, tmp_path):
    import json
    ws, audio = populated_workspace
    out = export_bundle(ws, audio, tmp_path / 'portable')
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['stems']['vocals'] == 'stems/vocals.wav'
    assert manifest['midis']['all_stems'] == 'midis/all_stems.mid'


def test_profile_mismatch_rejected(populated_workspace, tmp_path):
    ws, audio = populated_workspace
    with pytest.raises(ValueError, match='profile'):
        export_bundle(ws, audio, tmp_path / 'wrong', profile='fast')


def test_export_never_merges_into_existing_directory(populated_workspace, tmp_path):
    ws, audio = populated_workspace
    out = tmp_path / 'existing'
    out.mkdir()
    (out / 'keep').write_text('user file')
    with pytest.raises(ValueError, match='exists'):
        export_bundle(ws, audio, out)
    assert (out / 'keep').read_text() == 'user file'


@pytest.mark.parametrize('problem', ['corrupt_midi', 'incomplete_stage', 'bad_state'])
def test_invalid_bundle_content_rejected(populated_workspace, tmp_path, problem):
    ws, audio = populated_workspace
    m = ws.get_manifest(audio)
    if problem == 'corrupt_midi':
        Path(m.midis['vocals']).write_bytes(b'bad')
    elif problem == 'incomplete_stage':
        m.completed_stages.remove('structure')
    else:
        m.transcription_results['vocals']['status'] = 'notes'
    ws.save_manifest(audio, m)
    with pytest.raises(ValueError, match='Incomplete'):
        export_bundle(ws, audio, tmp_path / 'invalid')


def test_stale_raw_midi_is_not_exported(populated_workspace, tmp_path):
    ws, audio = populated_workspace
    (ws.get_track_dir(audio) / 'midis' / 'guitar_raw.mid').write_bytes(b'old')
    out = export_bundle(ws, audio, tmp_path / 'clean')
    assert not (out / 'midis' / 'guitar_raw.mid').exists()


def test_zip_suffix_cannot_overwrite_source(tmp_path):
    ws = WorkspaceManager(tmp_path / 'cache')
    source = tmp_path / 'song.zip'
    source.write_bytes(b'user input')
    with pytest.raises(ValueError, match='overwrite source'):
        export_bundle(ws, source, tmp_path / 'song.any', archive=True)
    assert source.read_bytes() == b'user input'


def test_exports_are_stable_across_python_hash_seeds(complete_workspace, tmp_path):
    import os
    import subprocess
    import sys
    ws, source = complete_workspace
    outputs = [tmp_path / 'seed1', tmp_path / 'seed2']
    for seed, out in zip(['1', '2'], outputs):
        code = ('from music_re.workspace import WorkspaceManager; '
                'from music_re.core.export import export_bundle; '
                f'export_bundle(WorkspaceManager({str(ws.cache_root)!r}), {str(source)!r}, {str(out)!r})')
        subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONHASHSEED': seed}, check=True)
    assert (outputs[0] / 'manifest.json').read_bytes() == (outputs[1] / 'manifest.json').read_bytes()
