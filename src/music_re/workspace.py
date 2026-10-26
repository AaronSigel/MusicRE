import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


def compute_file_content_hash(path: Union[str, Path], chunk_size: int = 64 * 1024) -> str:
    """Compute 16-character SHA256 hex digest by streaming the entire file."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()[:16]


def compute_file_hash(path: Union[str, Path]) -> str:
    """Backwards-compatible wrapper returning content hash."""
    return compute_file_content_hash(path)


class WorkspaceManager:
    """Manages cache workspace directories indexed by content hash."""

    def __init__(self, cache_root: Optional[Union[str, Path]] = None):
        if cache_root is None:
            self.cache_root = Path.home() / ".music_re_cache"
        else:
            self.cache_root = Path(cache_root).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def get_track_dir(self, file_path: Union[str, Path]) -> Path:
        p = Path(file_path).resolve()
        file_hash = compute_file_hash(p)
        stem_name = p.stem.replace(" ", "_")
        track_dir = self.cache_root / f"{stem_name}_{file_hash}"
        track_dir.mkdir(parents=True, exist_ok=True)
        return track_dir

    def get_stems_dir(self, file_path: Union[str, Path], model_name: Optional[str] = None) -> Path:
        base_stems_dir = self.get_track_dir(file_path) / "stems"
        if model_name:
            stems_dir = base_stems_dir / model_name
        else:
            stems_dir = base_stems_dir
        stems_dir.mkdir(parents=True, exist_ok=True)
        return stems_dir

    def get_stem_path(
        self,
        file_path: Union[str, Path],
        stem_name: str,
        ext: str = "wav",
        model_name: Optional[str] = None,
    ) -> Path:
        return self.get_stems_dir(file_path, model_name=model_name) / f"{stem_name}.{ext}"

    def save_metadata(self, file_path: Union[str, Path], metadata: Dict[str, Any]) -> Path:
        meta_file = self.get_track_dir(file_path) / "metadata.json"
        tmp_file = meta_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        tmp_file.replace(meta_file)
        return meta_file

    def load_metadata(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        meta_file = self.get_track_dir(file_path) / "metadata.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
