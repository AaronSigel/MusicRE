import io
import os
import sys
import tarfile
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.fetch_slakh_subset import fetch_babyslakh


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _build_fake_babyslakh_tar(tmp_path: Path, track_ids) -> Path:
    archive = tmp_path / "fake_babyslakh.tar.gz"
    with tarfile.open(archive, mode="w:gz") as tf:
        for track_id in track_ids:
            _add_bytes(tf, f"{track_id}/metadata.yaml", b"stems: {}\n")
            _add_bytes(tf, f"{track_id}/MIDI/S00.mid", b"\x00")
            _add_bytes(tf, f"{track_id}/stems/S00.wav", b"\x00")
            _add_bytes(tf, f"{track_id}/mix.wav", b"\x00")
    return archive


def test_fetch_babyslakh_extracts_only_requested_tracks(tmp_path):
    fixture_tar = _build_fake_babyslakh_tar(tmp_path, track_ids=["Track00001", "Track00002"])
    dest = tmp_path / "dest"

    tracks = fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=fixture_tar)

    assert [t.name for t in tracks] == ["Track00001"]
    assert (dest / "raw" / "Track00001" / "metadata.yaml").exists()
    assert not (dest / "raw" / "Track00002").exists()


def test_fetch_babyslakh_is_idempotent(tmp_path):
    fixture_tar = _build_fake_babyslakh_tar(tmp_path, track_ids=["Track00001"])
    dest = tmp_path / "dest"
    fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=fixture_tar)
    tracks_again = fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=fixture_tar)
    assert len(tracks_again) == 1


def test_fetch_babyslakh_rejects_corrupt_archive(tmp_path):
    bad_tar = tmp_path / "bad.tar.gz"
    bad_tar.write_bytes(b"not a real tarball")
    dest = tmp_path / "dest"

    with pytest.raises(Exception):
        fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=bad_tar)
    assert not (dest / "raw" / "Track00001").exists()  # Review Focus #4


@pytest.mark.dataset
def test_fetch_babyslakh_real_download(tmp_path):
    if not os.environ.get('MUSIC_RE_TEST_SLAKH'):
        pytest.skip('Set MUSIC_RE_TEST_SLAKH=1 to download the real 882.8 MB BabySlakh archive.')
    tracks = fetch_babyslakh(tmp_path / "dest", track_ids=["Track00001"])
    assert tracks and (tracks[0] / "metadata.yaml").exists()
