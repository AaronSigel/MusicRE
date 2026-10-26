"""Describe selected local checkpoints without downloading or loading weights."""
import hashlib
from importlib.metadata import distribution
from pathlib import Path

import torch
import yaml


def _describe(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(id=path.name, sha256=digest)


def model_inventory(separator_model, structure_model):
    result = dict(separator=[], structure=[], tonal=[])
    demucs = distribution('demucs')
    # Demucs 4.1 prefers Hugging Face safetensors; older releases use
    # torch.hub .th files. Record the preferred available representation.
    from huggingface_hub import try_to_load_from_cache
    from demucs.hf import hf_repo_name, DEFAULT_NAMESPACE
    repo = f'{DEFAULT_NAMESPACE}/{hf_repo_name(separator_model)}'
    cached_definition = try_to_load_from_cache(repo, f'{separator_model}.yaml')
    if isinstance(cached_definition, str):
        result['separator'].append(_describe(cached_definition))
        for signature in yaml.safe_load(Path(cached_definition).read_text())['models']:
            cached = try_to_load_from_cache(repo, f'{signature}.safetensors')
            if isinstance(cached, str):
                result['separator'].append(_describe(cached))
    definition = Path(demucs.locate_file(f'demucs/remote/{separator_model}.yaml'))
    if not result['separator'] and definition.is_file():
        signatures = yaml.safe_load(definition.read_text())['models']
        for signature in signatures:
            for path in (Path(torch.hub.get_dir()) / 'checkpoints').glob(f'{signature}-*.th'):
                result['separator'].append(_describe(path))
    if structure_model:
        from huggingface_hub import try_to_load_from_cache
        from allin1_infer.models.loaders import NAME_TO_FILE, ENSEMBLE_MODELS
        for model in ENSEMBLE_MODELS.get(structure_model, [structure_model]):
            cached = try_to_load_from_cache('taejunkim/allinone', NAME_TO_FILE[model])
            if isinstance(cached, str):
                result['structure'].append(_describe(cached))
    pitch = Path(distribution('basic-pitch').locate_file('basic_pitch/saved_models/icassp_2022/nmp.onnx'))
    if pitch.is_file():
        result['tonal'].append(_describe(pitch))
    return result
