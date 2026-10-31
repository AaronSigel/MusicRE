"""Stage 0: pin pipeline STAGES list and run_pipeline signature."""
import inspect

import pytest


def test_pipeline_stages_exact():
    from music_re.core.pipeline import STAGES

    assert STAGES == ['separation', 'analysis', 'structure', 'transcription', 'spectral_validation', 'evaluation']


def test_run_pipeline_param_names():
    from music_re.core.pipeline import run_pipeline

    params = list(inspect.signature(run_pipeline).parameters.keys())
    assert params == [
        'file_path', 'workspace', 'six_stems', 'device',
        'profile', 'progress_callback', 'separator_backend', 'spectral_validation',
    ]
    assert [parameter.default for parameter in inspect.signature(run_pipeline).parameters.values()] == [
        inspect.Parameter.empty, None, False, 'auto',
        'balanced', None, 'demucs', 'off',
    ]
