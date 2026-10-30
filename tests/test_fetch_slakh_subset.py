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


def _build_fake_babyslakh_tar(tmp_path: Path, track_ids, top_level: str = "babyslakh_16k") -> Path:
    """Mirror the real archive: everything sits under one top-level folder
    (verified against the live Zenodo archive: babyslakh_16k/TrackXXXXX/...)."""
    archive = tmp_path / "fake_babyslakh.tar.gz"
    with tarfile.open(archive, mode="w:gz") as tf:
        for track_id in track_ids:
            prefix = f"{top_level}/{track_id}" if top_level else track_id
            _add_bytes(tf, f"{prefix}/metadata.yaml", b"stems: {}\n")
            _add_bytes(tf, f"{prefix}/MIDI/S00.mid", b"\x00")
            _add_bytes(tf, f"{prefix}/stems/S00.wav", b"\x00")
            _add_bytes(tf, f"{prefix}/mix.wav", b"\x00")
    return archive


def test_fetch_babyslakh_extracts_only_requested_tracks(tmp_path):
    fixture_tar = _build_fake_babyslakh_tar(tmp_path, track_ids=["Track00001", "Track00002"])
    dest = tmp_path / "dest"

    tracks = fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=fixture_tar)

    assert [t.name for t in tracks] == ["Track00001"]
    assert (dest / "raw" / "Track00001" / "metadata.yaml").exists()
    assert not (dest / "raw" / "Track00002").exists()


def test_fetch_babyslakh_strips_the_archive_top_level_folder(tmp_path):
    # Review C1: the real BabySlakh archive nests every track under one
    # top-level folder (babyslakh_16k/TrackXXXXX/...), not at the archive
    # root. A naive `startswith(f"{track_id}/")` filter matches nothing.
    fixture_tar = _build_fake_babyslakh_tar(tmp_path, track_ids=["Track00001"])
    dest = tmp_path / "dest"

    tracks = fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=fixture_tar)

    assert tracks == [dest / "raw" / "Track00001"]
    assert (dest / "raw" / "Track00001" / "metadata.yaml").exists()
    assert not (dest / "raw" / "babyslakh_16k").exists()


def test_fetch_babyslakh_raises_for_a_track_id_not_in_the_archive(tmp_path):
    fixture_tar = _build_fake_babyslakh_tar(tmp_path, track_ids=["Track00001"])
    dest = tmp_path / "dest"

    with pytest.raises(ValueError, match="Track00099"):
        fetch_babyslakh(dest, track_ids=["Track00099"], archive_path=fixture_tar)
    assert not (dest / "raw" / "Track00099").exists()


def test_fetch_babyslakh_is_idempotent(tmp_path):
    fixture_tar = _build_fake_babyslakh_tar(tmp_path, track_ids=["Track00001"])
    dest = tmp_path / "dest"
    fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=fixture_tar)
    tracks_again = fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=fixture_tar)
    assert len(tracks_again) == 1


def test_fetch_babyslakh_download_is_atomic_on_failure(tmp_path, monkeypatch):
    # Review I3: urlretrieve wrote straight to the final archive path, so
    # an interrupted 882 MB download left a corrupt archive that every
    # later run would reuse (and fail on) forever.
    dest = tmp_path / "dest"

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(b"partial-garbage")
        raise OSError("connection reset")

    monkeypatch.setattr("scripts.fetch_slakh_subset.urllib.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(OSError):
        fetch_babyslakh(dest, track_ids=["Track00001"])

    final_archive = dest / "babyslakh_16k.tar.gz"
    assert not final_archive.exists()
    assert not any(dest.glob("*.part"))


def test_fetch_babyslakh_download_succeeds_lands_at_final_path(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    real_archive = _build_fake_babyslakh_tar(tmp_path, track_ids=["Track00001"])

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(real_archive.read_bytes())

    monkeypatch.setattr("scripts.fetch_slakh_subset.urllib.request.urlretrieve", fake_urlretrieve)
    # The download is size-checked against the real archive; match the fixture.
    monkeypatch.setattr("scripts.fetch_slakh_subset.BABYSLAKH_SIZE", real_archive.stat().st_size)

    tracks = fetch_babyslakh(dest, track_ids=["Track00001"])

    final_archive = dest / "babyslakh_16k.tar.gz"
    assert final_archive.exists()
    assert not any(dest.glob("*.part"))
    assert tracks == [dest / "raw" / "Track00001"]


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
