#!/usr/bin/env python3
"""Fetch a small subset of BabySlakh (CC-BY-4.0, Manilow et al. 2019,
https://zenodo.org/records/4603870) for the Slakh benchmark baseline.

BabySlakh ships as one 882.8 MB tar.gz covering all 20 tracks; this module
downloads that archive once and extracts only the requested track ids.

Extraction is staged: members are validated, extracted into a private
``raw/.staging-*`` directory, and each track is published with an atomic
``os.replace`` only once everything extracted cleanly. Any failure removes
only the staging directory, so pre-existing ``raw/<track>`` data survives.
"""
import copy
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

BABYSLAKH_URL = "https://zenodo.org/api/records/4603870/files/babyslakh_16k.tar.gz/content"
BABYSLAKH_SIZE = 882818115

DEFAULT_TRACK_IDS = ["Track00001", "Track00002", "Track00003"]


def _download(archive_path: Path) -> None:
    # Download to a .part file and only move it into place on success, so an
    # interrupted or truncated transfer never leaves a corrupt archive at the
    # path every later run reuses.
    partial_path = Path(str(archive_path) + ".part")
    try:
        urllib.request.urlretrieve(BABYSLAKH_URL, partial_path)
        size = partial_path.stat().st_size
        if size != BABYSLAKH_SIZE:
            raise ValueError(
                f"Downloaded BabySlakh archive has size {size}, expected {BABYSLAKH_SIZE}"
            )
        partial_path.replace(archive_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def _select_members(
    members: List[tarfile.TarInfo], track_id: str, archive_path: Path
) -> List[tarfile.TarInfo]:
    """Return re-rooted copies of the members belonging to track_id.

    The real archive nests every track under one top-level folder
    (babyslakh_16k/TrackXXXXX/...), so match on the track id as a path
    component and re-root at the track id. Raises ValueError for any
    member type or name that could escape or alias outside staging.
    """
    selected = []
    for m in members:
        parts = PurePosixPath(m.name).parts
        if track_id not in parts:
            continue
        if not (m.isreg() or m.isdir()):
            raise ValueError(f"Refusing non-regular archive member {m.name!r} (type {m.type!r})")
        if m.name.startswith("/") or PurePosixPath(m.name).is_absolute():
            raise ValueError(f"Refusing absolute archive member {m.name!r}")
        if ".." in parts:
            raise ValueError(f"Refusing archive member with '..': {m.name!r}")
        rerooted = PurePosixPath(*parts[parts.index(track_id):])
        if ".." in rerooted.parts or rerooted.is_absolute():
            raise ValueError(f"Refusing archive member {m.name!r}")
        clone = copy.copy(m)
        clone.name = str(rerooted)
        selected.append(clone)
    if not selected:
        raise ValueError(f"{track_id} not found in archive {archive_path}")
    return selected


def _check_inside(staging: Path, members: List[tarfile.TarInfo]) -> None:
    root = staging.resolve()
    for m in members:
        target = (root / m.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Archive member {m.name!r} escapes the staging directory")


def _publish(staging: Path, raw_dir: Path, track_id: str, force: bool) -> None:
    new_dir = staging / track_id
    final_dir = raw_dir / track_id
    if not final_dir.exists():
        os.replace(new_dir, final_dir)
        return
    if not force:
        return
    old_dir = staging / f".old-{track_id}"
    os.replace(final_dir, old_dir)
    try:
        os.replace(new_dir, final_dir)
    except BaseException:
        os.replace(old_dir, final_dir)
        raise
    shutil.rmtree(old_dir, ignore_errors=True)


def fetch_babyslakh(
    dest_dir: Path,
    track_ids: Optional[List[str]] = None,
    archive_path: Optional[Path] = None,
    force: bool = False,
) -> List[Path]:
    """Download (if needed) and extract the requested BabySlakh track ids.

    Returns the sorted list of extracted track directories under
    dest_dir/raw/.
    """
    if track_ids is None:
        track_ids = DEFAULT_TRACK_IDS
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = dest_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not force and all((raw_dir / t).is_dir() for t in track_ids):
        return sorted(raw_dir / t for t in track_ids)

    if archive_path is None:
        archive_path = dest_dir / "babyslakh_16k.tar.gz"
        if force or not Path(archive_path).exists():
            _download(Path(archive_path))
    archive_path = Path(archive_path)

    todo = [t for t in track_ids if force or not (raw_dir / t).is_dir()]
    staging = Path(tempfile.mkdtemp(dir=raw_dir, prefix=".staging-"))
    try:
        try:
            with tarfile.open(archive_path, "r:gz") as tf:
                members = tf.getmembers()
                selected: Dict[str, List[tarfile.TarInfo]] = {
                    t: _select_members(members, t, archive_path) for t in todo
                }
                for track_id in todo:
                    _check_inside(staging, selected[track_id])
                    tf.extractall(staging, members=selected[track_id], filter="data")
                    if not (staging / track_id).is_dir():
                        raise ValueError(f"{track_id} did not extract as a directory")
        except tarfile.TarError as e:
            raise tarfile.TarError(
                f"Corrupt or incomplete archive at {archive_path}: {e}. Delete it and retry."
            ) from e
        for track_id in todo:
            _publish(staging, raw_dir, track_id, force)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return sorted(raw_dir / t for t in track_ids)
