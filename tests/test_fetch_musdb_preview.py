import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import fetch_musdb_preview as fm  # noqa: E402
from scripts.run_quality_benchmarks import track_in_split  # noqa: E402

NAMES = [f'MUSDB18-7/{subset}/Artist {i} - Song (x).stem.mp4'
         for subset in ('train', 'test') for i in range(40)] + ['MUSDB18-7/', 'MUSDB18-7/readme.txt']


def test_clip_id_is_safe_and_keeps_subset():
    cid = fm.clip_id('MUSDB18-7/train/A Band - Song (Live).stem.mp4')
    assert cid == 'train__A_Band_-_Song_Live'
    assert '/' not in cid and ' ' not in cid


def test_subset_is_deterministic_balanced_and_bounded():
    chosen = fm.select_subset(NAMES, limit=50)
    assert chosen == fm.select_subset(list(reversed(NAMES)), limit=50)
    assert len(chosen) == 50
    assert all(n.endswith('.stem.mp4') for n in chosen)
    tune = [n for n in chosen if track_in_split(fm.clip_id(n), 'tune')]
    assert len(tune) == 25
    assert len(fm.select_subset(NAMES[:4], limit=50)) == 4
