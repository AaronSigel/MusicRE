import os
from pathlib import Path

import pytest


@pytest.mark.real_e2e
def test_real_mp3_acceptance(tmp_path):
    source = os.environ.get('MUSIC_RE_TEST_MP3')
    if not source:
        pytest.skip('Set MUSIC_RE_TEST_MP3 to an authorized real recording; models are not mocked.')
    from scripts.verify_real_e2e import verify
    report = verify(Path(source), tmp_path / 'acceptance')
    assert report['status'] == 'passed'
