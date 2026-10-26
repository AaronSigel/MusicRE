from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from music_re.core.audio_io import decode_audio
from music_re.core.analyzer import analyze_audio
from music_re.core.separator import separate_stems
from music_re.core.midi_tonal import transcribe_tonal
from music_re.workspace import WorkspaceManager

@pytest.fixture
def multi_stem_audio(tmp_path: Path):
    sr = 22050
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Tone + noise
    tone = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    noise = np.random.uniform(-0.05, 0.05, len(t))
    mix = (tone + noise).astype(np.float32)

    wav_p = tmp_path / "song.wav"
    flac_p = tmp_path / "song.flac"
    sf.write(str(wav_p), mix, sr, subtype="PCM_16")
    sf.write(str(flac_p), mix, sr, format="FLAC")
    return wav_p, flac_p, sr

def test_stage1_multi_format_decoding(multi_stem_audio):
    wav_p, flac_p, sr = multi_stem_audio

    # Test WAV decoding
    y_wav, sr_wav = decode_audio(wav_p, target_sr=22050, normalize_db=-1.0)
    assert sr_wav == 22050
    assert y_wav.dtype == np.float32
    assert 0.88 <= np.max(np.abs(y_wav)) <= 0.90

    # Test FLAC decoding
    y_flac, sr_flac = decode_audio(flac_p, target_sr=22050, normalize_db=-1.0)
    assert sr_flac == 22050
    assert y_flac.dtype == np.float32
    assert 0.88 <= np.max(np.abs(y_flac)) <= 0.90

def test_stage1_e2e_workspace_pipeline(tmp_path: Path, seed_stem_cache):
    sr = 22050
    duration_sec = 3.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    track_p = tmp_path / "e2e_track.wav"
    sf.write(str(track_p), y, sr)

    ws = WorkspaceManager(cache_root=tmp_path / "cache")
    track_dir = ws.get_track_dir(track_p)

    # 1. Manifest starts in pending
    manifest = ws.get_manifest(track_p)
    assert manifest.status == "pending"

    # 2. Simulate separation stage with mock stems in segregated directory
    stems_dir = ws.get_stems_dir(track_p, model_name="htdemucs")
    mock_stems = {}
    for name in ["vocals", "drums", "bass", "other"]:
        stem_f = stems_dir / f"{name}.wav"
        sf.write(str(stem_f), y, sr)
        mock_stems[name] = str(stem_f)
    seed_stem_cache(track_p, mock_stems)

    sep_res = separate_stems(track_p, workspace=ws, six_stems=False)
    assert sep_res.model_used == "htdemucs"
    ws.mark_stage_complete(track_p, "separation", artifacts=sep_res.stems)

    # 3. Analyze audio (drums skips key detection, mix gets key)
    mix_analysis = analyze_audio(track_p)
    drum_analysis = analyze_audio(sep_res.stems["drums"], stem_name="drums")
    assert drum_analysis.key in ("N/A", None)
    assert drum_analysis.key_confidence == 0.0
    assert mix_analysis.key != "N/A"
    ws.mark_stage_complete(track_p, "analysis", artifacts=mix_analysis.model_dump())

    # 4. Transcribe tonal stem and verify dual MIDI files (quantized + raw)
    midis_dir = track_dir / "midis"
    midis_dir.mkdir(parents=True, exist_ok=True)
    out_midi = midis_dir / "vocals.mid"
    raw_midi = midis_dir / "vocals_raw.mid"

    res_midi = transcribe_tonal(sep_res.stems["vocals"], out_midi, bpm=120.0, save_raw_copy=True)
    assert out_midi.exists()
    assert raw_midi.exists()

    ws.mark_stage_complete(track_p, "transcription", artifacts={"vocals": str(out_midi)})
    ws.update_manifest(track_p, status="completed", current_stage="completed")

    final_manifest = ws.get_manifest(track_p)
    assert final_manifest.status == "completed"
    assert "separation" in final_manifest.completed_stages
    assert "analysis" in final_manifest.completed_stages
    assert "transcription" in final_manifest.completed_stages
    assert len(final_manifest.stage_errors) == 0
