import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


def compute_file_hash(path: Union[str, Path]) -> str:
    """Compute 8-character SHA256 hex digest from first 64KB + file size."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}")

    hasher = hashlib.sha256()
    size = p.stat().st_size
    hasher.update(str(size).encode("utf-8"))

    with open(p, "rb") as f:
        chunk = f.read(64 * 1024)
        hasher.update(chunk)

    return hasher.hexdigest()[:8]


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

    def get_stems_dir(self, file_path: Union[str, Path]) -> Path:
        stems_dir = self.get_track_dir(file_path) / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        return stems_dir

    def get_stem_path(self, file_path: Union[str, Path], stem_name: str, ext: str = "wav") -> Path:
        return self.get_stems_dir(file_path) / f"{stem_name}.{ext}"

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
