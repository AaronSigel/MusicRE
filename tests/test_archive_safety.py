import io
import sys
import tarfile
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.fetch_slakh_subset as fss
from scripts.fetch_slakh_subset import fetch_babyslakh

TOP = "babyslakh_16k"


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _add_track(tf: tarfile.TarFile, track_id: str, payload: bytes = b"new"):
    prefix = f"{TOP}/{track_id}"
    _add_bytes(tf, f"{prefix}/metadata.yaml", b"stems: {}\n")
    _add_bytes(tf, f"{prefix}/mix.wav", payload)


def _build(tmp_path: Path, track_ids, extra=None, name="fake.tar.gz") -> Path:
    archive = tmp_path / name
    with tarfile.open(archive, mode="w:gz") as tf:
        for track_id in track_ids:
            _add_track(tf, track_id)
        if extra is not None:
            extra(tf)
    return archive


def _snapshot(root: Path):
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def _no_staging(dest: Path):
    raw = dest / "raw"
    return not raw.exists() or not any(raw.glob(".staging-*"))


def test_url_and_size_constants():
    assert fss.BABYSLAKH_URL == (
        "https://zenodo.org/api/records/4603870/files/babyslakh_16k.tar.gz/content"
    )
    assert fss.BABYSLAKH_SIZE == 882818115


def test_download_with_wrong_size_is_refused(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    real = _build(tmp_path, ["Track00001"])

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(real.read_bytes())

    monkeypatch.setattr(fss.urllib.request, "urlretrieve", fake_urlretrieve)
    with pytest.raises(ValueError, match="size"):
        fetch_babyslakh(dest, track_ids=["Track00001"])
    assert not (dest / "babyslakh_16k.tar.gz").exists()
    assert not any(dest.glob("*.part"))
    assert not (dest / "raw" / "Track00001").exists()


def _dotdot(tf):
    _add_bytes(tf, f"{TOP}/Track00001/../../evil.txt", b"evil")


def _leading_dotdot(tf):
    _add_bytes(tf, f"../Track00001/evil.txt", b"evil")


def _absolute(tf):
    _add_bytes(tf, f"/{TOP}/Track00001/abs.txt", b"evil")


def _symlink(tf):
    info = tarfile.TarInfo(name=f"{TOP}/Track00001/link")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"
    tf.addfile(info)


def _hardlink(tf):
    info = tarfile.TarInfo(name=f"{TOP}/Track00001/hard")
    info.type = tarfile.LNKTYPE
    info.linkname = f"{TOP}/Track00001/mix.wav"
    tf.addfile(info)


def _device(tf):
    info = tarfile.TarInfo(name=f"{TOP}/Track00001/dev")
    info.type = tarfile.CHRTYPE
    info.devmajor, info.devminor = 1, 3
    tf.addfile(info)


@pytest.mark.parametrize(
    "extra", [_dotdot, _leading_dotdot, _absolute, _symlink, _hardlink, _device],
    ids=["dotdot", "leading-dotdot", "absolute", "symlink", "hardlink", "device"],
)
def test_malicious_member_raises_and_publishes_nothing(tmp_path, extra):
    work = tmp_path / "work"
    work.mkdir()
    archive = _build(work, ["Track00001", "Track00002"], extra=extra)
    dest = tmp_path / "dest"
    before_outside = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="Refusing"):
        fetch_babyslakh(dest, track_ids=["Track00001", "Track00002"], archive_path=archive)

    assert not (dest / "raw" / "Track00001").exists()
    assert not (dest / "raw" / "Track00002").exists()
    assert not (tmp_path / "evil.txt").exists()
    assert not (dest / "evil.txt").exists()
    assert _no_staging(dest)
    after = _snapshot(tmp_path)
    added = set(after) - set(before_outside)
    assert added <= {"dest", "dest/raw"}


@pytest.mark.parametrize("force", [False, True])
def test_corrupt_archive_keeps_pre_existing_user_data(tmp_path, force):
    dest = tmp_path / "dest"
    track = dest / "raw" / "Track00001"
    track.mkdir(parents=True)
    (track / "user-data.txt").write_bytes(b"precious\x00bytes")
    before = _snapshot(dest / "raw")
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"not a real tarball")

    with pytest.raises(tarfile.TarError):
        fetch_babyslakh(
            dest, track_ids=["Track00001", "Track00002"], archive_path=bad, force=force
        )

    assert _snapshot(dest / "raw") == before
    assert _no_staging(dest)


def test_force_with_malicious_archive_keeps_existing_track(tmp_path):
    dest = tmp_path / "dest"
    track = dest / "raw" / "Track00001"
    track.mkdir(parents=True)
    (track / "user-data.txt").write_bytes(b"precious")
    before = _snapshot(dest / "raw")
    work = tmp_path / "work"
    work.mkdir()
    archive = _build(work, ["Track00001"], extra=_symlink)

    with pytest.raises(ValueError, match="Refusing non-regular"):
        fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=archive, force=True)

    assert _snapshot(dest / "raw") == before
    assert _no_staging(dest)


def test_force_failure_mid_publish_restores_old_track(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    track = dest / "raw" / "Track00001"
    track.mkdir(parents=True)
    (track / "user-data.txt").write_bytes(b"precious")
    before = _snapshot(dest / "raw")
    archive = _build(tmp_path, ["Track00001"])

    real_replace = fss.os.replace

    def flaky_replace(src, dst):
        if Path(dst) == track and ".staging-" in str(src) and Path(src).name == "Track00001":
            raise OSError("simulated publish failure")
        return real_replace(src, dst)

    monkeypatch.setattr(fss.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated"):
        fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=archive, force=True)

    assert _snapshot(dest / "raw") == before
    assert _no_staging(dest)


def test_existing_tracks_without_force_never_open_archive(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    for tid in ["Track00001", "Track00002"]:
        (dest / "raw" / tid).mkdir(parents=True)

    def boom(*a, **k):
        raise AssertionError("archive must not be opened or downloaded")

    monkeypatch.setattr(fss.tarfile, "open", boom)
    monkeypatch.setattr(fss.urllib.request, "urlretrieve", boom)

    tracks = fetch_babyslakh(
        dest,
        track_ids=["Track00002", "Track00001"],
        archive_path=tmp_path / "does-not-exist.tar.gz",
    )
    assert tracks == [dest / "raw" / "Track00001", dest / "raw" / "Track00002"]

    tracks = fetch_babyslakh(dest, track_ids=["Track00001"])
    assert tracks == [dest / "raw" / "Track00001"]


def test_happy_path_publishes_requested_tracks_only(tmp_path):
    archive = _build(tmp_path, ["Track00001", "Track00002", "Track00003"])
    dest = tmp_path / "dest"

    tracks = fetch_babyslakh(dest, track_ids=["Track00003", "Track00001"], archive_path=archive)

    assert tracks == [dest / "raw" / "Track00001", dest / "raw" / "Track00003"]
    assert (dest / "raw" / "Track00001" / "mix.wav").read_bytes() == b"new"
    assert (dest / "raw" / "Track00003" / "metadata.yaml").exists()
    assert not (dest / "raw" / "Track00002").exists()
    assert sorted(p.name for p in (dest / "raw").iterdir()) == ["Track00001", "Track00003"]
    assert _no_staging(dest)


def test_force_replaces_existing_track(tmp_path):
    dest = tmp_path / "dest"
    track = dest / "raw" / "Track00001"
    track.mkdir(parents=True)
    (track / "stale.txt").write_bytes(b"old")
    archive = _build(tmp_path, ["Track00001"])

    tracks = fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=archive, force=True)

    assert tracks == [track]
    assert not (track / "stale.txt").exists()
    assert (track / "mix.wav").read_bytes() == b"new"
    assert _no_staging(dest)


def test_missing_track_id_leaves_no_staging(tmp_path):
    archive = _build(tmp_path, ["Track00001"])
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="Track00099"):
        fetch_babyslakh(dest, track_ids=["Track00001", "Track00099"], archive_path=archive)
    assert not (dest / "raw" / "Track00001").exists()
    assert _no_staging(dest)


def test_force_restore_failure_keeps_old_track_outside_staging(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    raw = dest / "raw"
    track = raw / "Track00001"
    track.mkdir(parents=True)
    (track / "user-data.txt").write_bytes(b"precious")
    archive = _build(tmp_path, ["Track00001"])

    real_replace = fss.os.replace

    def failing_replace(src, dst):
        # Moving the old copy aside succeeds; publishing the new copy and
        # restoring the old one both fail.
        if Path(dst) == track:
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(fss.os, "replace", failing_replace)
    with pytest.raises(OSError) as excinfo:
        fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=archive, force=True)

    survivors = [p for p in raw.rglob("user-data.txt") if p.read_bytes() == b"precious"]
    assert len(survivors) == 1
    assert str(survivors[0].parent) in str(excinfo.value)
    assert _no_staging(dest)


def test_track_id_deeper_than_one_folder_is_ignored(tmp_path):
    archive = tmp_path / "deep.tar.gz"
    with tarfile.open(archive, mode="w:gz") as tf:
        _add_bytes(tf, "a/b/Track00001/mix.wav", b"deep")
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="Track00001 not found"):
        fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=archive)
    assert not (dest / "raw" / "Track00001").exists()
    assert _no_staging(dest)


def test_track_under_two_top_level_folders_is_refused(tmp_path):
    archive = tmp_path / "two.tar.gz"
    with tarfile.open(archive, mode="w:gz") as tf:
        _add_bytes(tf, "a/Track00001/x", b"x")
        _add_bytes(tf, "b/Track00001/y", b"y")
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="more than one"):
        fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=archive)
    assert not (dest / "raw" / "Track00001").exists()
    assert _no_staging(dest)


@pytest.mark.parametrize("bad_id", ["", ".", "..", "a/b", "a\\b", "/abs", "../Track00001"])
def test_invalid_track_ids_are_rejected(tmp_path, bad_id):
    archive = _build(tmp_path, ["Track00001"])
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="track id"):
        fetch_babyslakh(dest, track_ids=[bad_id], archive_path=archive)
    assert not (dest / "raw").exists() or not any((dest / "raw").iterdir())


@pytest.mark.parametrize("force", [False, True])
def test_existing_track_path_that_is_a_file_raises(tmp_path, force):
    dest = tmp_path / "dest"
    (dest / "raw").mkdir(parents=True)
    (dest / "raw" / "Track00001").write_bytes(b"user file")
    archive = _build(tmp_path, ["Track00001"])
    with pytest.raises(ValueError, match="not a directory"):
        fetch_babyslakh(dest, track_ids=["Track00001"], archive_path=archive, force=force)
    assert (dest / "raw" / "Track00001").read_bytes() == b"user file"
    assert _no_staging(dest)


def test_extractall_uses_data_filter(tmp_path, monkeypatch):
    archive = _build(tmp_path, ["Track00001"])
    calls = []
    real = tarfile.TarFile.extractall

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("filter"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "extractall", spy)
    fetch_babyslakh(tmp_path / "dest", track_ids=["Track00001"], archive_path=archive)
    assert calls == ["data"]
