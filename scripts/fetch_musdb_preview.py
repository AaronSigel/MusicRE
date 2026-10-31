#!/usr/bin/env python3
"""Fetch the MUSDB18 7-second previews (Rafii et al. 2017,
https://zenodo.org/records/3270814) for spectral presence calibration.

Downloads MUSDB18-7-STEMS.zip (147 MB) into ``dest_dir`` (resume-safe: the
transfer goes to a ``.part`` file, resumed with an HTTP Range request and only
moved into place once complete), then extracts a deterministic subset of at
most ``limit`` clips, half from each ``track_in_split`` split. For every clip
the mixture (stream 0) and the vocals (stream 4) are written as WAV with
ffmpeg to ``dest_dir/clips/<clip_id>/{mixture,vocals}.wav``.

Usage:
    uv run --locked python scripts/fetch_musdb_preview.py --dest data/benchmarks/musdb7 --limit 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_quality_benchmarks import track_in_split  # noqa: E402

MUSDB7_URL = 'https://zenodo.org/records/3270814/files/MUSDB18-7-STEMS.zip?download=1'
MUSDB7_SIZE = 147209385
ARCHIVE_NAME = 'MUSDB18-7-STEMS.zip'
STEM_SUFFIX = '.stem.mp4'
# MUSDB stem order: 0 mixture, 1 drums, 2 bass, 3 other, 4 vocals.
STREAMS = {'mixture': 0, 'vocals': 4}


def clip_id(member_name: str) -> str:
    """Stable, filesystem-safe id: '<subset>__<title>' from '.../<subset>/<title>.stem.mp4'."""
    path = PurePosixPath(member_name)
    title = path.name[:-len(STEM_SUFFIX)]
    subset = path.parent.name or 'root'
    return re.sub(r'[^A-Za-z0-9._-]+', '_', f'{subset}__{title}').strip('_')


def select_subset(member_names, limit: int = 50) -> list[str]:
    """Deterministic subset: at most limit//2 clips per split, ordered by sha1 of the clip id."""
    stems = sorted(n for n in member_names if n.endswith(STEM_SUFFIX) and '__MACOSX' not in n)
    per_split = limit // 2
    chosen = []
    for split in ('tune', 'test'):
        names = [n for n in stems if track_in_split(clip_id(n), split)]
        names.sort(key=lambda n: hashlib.sha1(clip_id(n).encode()).hexdigest())
        chosen.extend(names[:per_split])
    return sorted(chosen)


def download(url: str, archive_path: Path, expected_size: int | None = MUSDB7_SIZE) -> Path:
    archive_path = Path(archive_path)
    if archive_path.is_file() and (expected_size is None or archive_path.stat().st_size == expected_size):
        return archive_path
    part = Path(str(archive_path) + '.part')
    offset = part.stat().st_size if part.is_file() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and response.status == 206
        with open(part, 'ab' if append else 'wb') as fh:
            shutil.copyfileobj(response, fh, length=1 << 20)
    size = part.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f'{part} has {size} bytes, expected {expected_size}; rerun to resume')
    part.replace(archive_path)
    return archive_path


def extract_wav(stem_file: Path, stream: int, out_path: Path) -> None:
    tmp = out_path.with_suffix('.tmp.wav')
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(stem_file),
                    '-map', f'0:{stream}', '-c:a', 'pcm_s16le', str(tmp)], check=True)
    tmp.replace(out_path)


def fetch_musdb_preview(dest_dir: Path, limit: int = 50, archive_path: Path | None = None) -> list[dict]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path(archive_path) if archive_path else download(MUSDB7_URL, dest_dir / ARCHIVE_NAME)
    clips_dir = dest_dir / 'clips'
    clips_dir.mkdir(exist_ok=True)
    entries = []
    with zipfile.ZipFile(archive_path) as zf, tempfile.TemporaryDirectory(dir=dest_dir, prefix='.staging-') as tmp:
        for name in select_subset(zf.namelist(), limit):
            cid = clip_id(name)
            out = clips_dir / cid
            paths = {key: out / f'{key}.wav' for key in STREAMS}
            if not all(p.is_file() for p in paths.values()):
                out.mkdir(exist_ok=True)
                staged = Path(tmp) / f'{cid}{STEM_SUFFIX}'
                with zf.open(name) as src, open(staged, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                for key, stream in STREAMS.items():
                    extract_wav(staged, stream, paths[key])
                staged.unlink()
            entries.append(dict(clip_id=cid, member=name,
                                split='tune' if track_in_split(cid, 'tune') else 'test',
                                **{f'{k}_path': str(p) for k, p in paths.items()}))
    (dest_dir / 'subset.json').write_text(json.dumps(entries, indent=2))
    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dest', type=Path, default=Path('data/benchmarks/musdb7'))
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--archive', type=Path, default=None)
    args = parser.parse_args(argv)
    entries = fetch_musdb_preview(args.dest, args.limit, args.archive)
    print(f'{len(entries)} clips in {args.dest / "clips"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
