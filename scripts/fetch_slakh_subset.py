#!/usr/bin/env python3
"""Fetch a small subset of BabySlakh (CC-BY-4.0, Manilow et al. 2019,
https://zenodo.org/records/4603870) for the Slakh benchmark baseline.

BabySlakh ships as one 882.8 MB tar.gz covering all 20 tracks; this module
downloads that archive once and extracts only the requested track ids.
"""
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Optional

BABYSLAKH_URL = "https://zenodo.org/records/4603870/files/babyslakh_16k.tar.gz?download=1"

DEFAULT_TRACK_IDS = ["Track00001", "Track00002", "Track00003"]


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

    if archive_path is None:
        archive_path = dest_dir / "babyslakh_16k.tar.gz"
        if force or not Path(archive_path).exists():
            # Download to a .part file and only move it into place on
            # success, so an interrupted 882 MB transfer never leaves a
            # corrupt archive at the path every later run reuses.
            partial_path = Path(str(archive_path) + ".part")
            try:
                urllib.request.urlretrieve(BABYSLAKH_URL, partial_path)
                partial_path.replace(archive_path)
            except Exception:
                partial_path.unlink(missing_ok=True)
                raise
    archive_path = Path(archive_path)

    extracted: List[Path] = []
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            members = tf.getmembers()
            for track_id in track_ids:
                track_dir = raw_dir / track_id
                if track_dir.is_dir() and not force:
                    extracted.append(track_dir)
                    continue
                # The real BabySlakh archive nests every track under one
                # top-level folder (e.g. babyslakh_16k/TrackXXXXX/...), not
                # at the archive root, so match on the track id as a path
                # component and re-root the extracted path at dest_dir/raw/.
                matching = []
                for m in members:
                    parts = Path(m.name).parts
                    if track_id in parts:
                        m.name = str(Path(*parts[parts.index(track_id):]))
                        matching.append(m)
                if not matching:
                    raise ValueError(f"{track_id} not found in archive {archive_path}")
                try:
                    tf.extractall(raw_dir, members=matching)
                except Exception:
                    if track_dir.exists():
                        shutil.rmtree(track_dir, ignore_errors=True)
                    raise
                extracted.append(track_dir)
    except tarfile.TarError as e:
        # A corrupt/partial archive must not leave any half-extracted track
        # directory that could pass for a complete one (Review Focus #4).
        for track_id in track_ids:
            track_dir = raw_dir / track_id
            if track_dir.exists():
                shutil.rmtree(track_dir, ignore_errors=True)
        raise tarfile.TarError(
            f"Corrupt or incomplete archive at {archive_path}: {e}. Delete it and retry."
        ) from e

    return sorted(extracted)
