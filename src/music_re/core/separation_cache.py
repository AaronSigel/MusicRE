"""Checksummed, configuration- and checkpoint-bound separation cache records."""
import json
from importlib.metadata import version
from pathlib import Path

import soundfile as sf

from music_re.core.provenance import model_inventory
from music_re.workspace import compute_file_content_hash


def cache_context(source, backend, shifts, overlap, seed=0):
    is_demucs = backend.backend_name == 'demucs'
    return dict(schema_version=2, input_hash=compute_file_content_hash(source),
                backend=backend.backend_name, model=backend.model_name, device=backend.device,
                shifts=shifts, overlap=overlap, seed=seed,
                versions={name: version(name) for name in ('demucs', 'torch', 'torchaudio')} if is_demucs else {},
                weights=model_inventory(backend.model_name, None)['separator'] if is_demucs else [])


def cache_is_valid(marker, stems, context):
    try:
        record = json.loads(Path(marker).read_text())
        if record['context'] != context or set(record['hashes']) != set(stems):
            return False
        for role, path in stems.items():
            info = sf.info(path)
            if info.frames <= 0 or record['hashes'][role] != compute_file_content_hash(path):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False


def save_cache_record(marker, stems, context):
    for path in stems.values():
        if sf.info(path).frames <= 0:
            raise ValueError(f'Invalid separation output: {path}')
    payload = dict(context=context, hashes={role: compute_file_content_hash(path) for role, path in stems.items()})
    marker = Path(marker)
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
    temporary.replace(marker)
