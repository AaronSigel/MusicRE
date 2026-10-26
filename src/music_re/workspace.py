import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from music_re.schemas import MusicalTimeline, QualityReport, WorkspaceManifest


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

    def get_timeline_path(self, file_path: Union[str, Path]) -> Path:
        return self.get_track_dir(file_path) / "timeline.json"

    def save_timeline(self, file_path: Union[str, Path], timeline: MusicalTimeline) -> Path:
        tl_path = self.get_timeline_path(file_path)
        tmp_file = tl_path.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(timeline.model_dump(), f, indent=2, ensure_ascii=False)
        tmp_file.replace(tl_path)
        self.mark_stage_complete(file_path, "structure", artifacts={"timeline": timeline.model_dump()})
        return tl_path

    def load_timeline(self, file_path: Union[str, Path]) -> Optional[MusicalTimeline]:
        tl_path = self.get_timeline_path(file_path)
        if not tl_path.exists():
            return None
        with open(tl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MusicalTimeline(**data)

    def get_quality_report_path(self, file_path: Union[str, Path]) -> Path:
        return self.get_track_dir(file_path) / "quality_report.json"

    def save_quality_report(self, file_path: Union[str, Path], report: QualityReport) -> Path:
        qr_path = self.get_quality_report_path(file_path)
        tmp_file = qr_path.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
        tmp_file.replace(qr_path)
        self.mark_stage_complete(file_path, "evaluation", artifacts={"report": report.model_dump()})
        return qr_path

    def load_quality_report(self, file_path: Union[str, Path]) -> Optional[QualityReport]:
        qr_path = self.get_quality_report_path(file_path)
        if not qr_path.exists():
            return None
        with open(qr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return QualityReport(**data)

    def load_metadata(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        meta_file = self.get_track_dir(file_path) / "metadata.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_manifest_path(self, file_path: Union[str, Path]) -> Path:
        return self.get_track_dir(file_path) / "manifest.json"

    def get_manifest(self, file_path: Union[str, Path]) -> WorkspaceManifest:
        m_path = self.get_manifest_path(file_path)
        if m_path.exists():
            with open(m_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return WorkspaceManifest(**data)

        p = Path(file_path).resolve()
        content_hash = compute_file_content_hash(p)
        manifest = WorkspaceManifest(
            track_id=content_hash,
            input_file=str(p),
            status="pending",
        )
        self.save_manifest(file_path, manifest)
        return manifest

    def save_manifest(self, file_path: Union[str, Path], manifest: WorkspaceManifest) -> Path:
        m_path = self.get_manifest_path(file_path)
        tmp_file = m_path.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2, ensure_ascii=False)
        tmp_file.replace(m_path)
        return m_path

    def update_manifest(self, file_path: Union[str, Path], **kwargs) -> WorkspaceManifest:
        manifest = self.get_manifest(file_path)
        for k, v in kwargs.items():
            if hasattr(manifest, k):
                setattr(manifest, k, v)
        self.save_manifest(file_path, manifest)
        return manifest

    def mark_stage_complete(
        self,
        file_path: Union[str, Path],
        stage: str,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> WorkspaceManifest:
        manifest = self.get_manifest(file_path)
        if stage not in manifest.completed_stages:
            manifest.completed_stages.append(stage)
        if artifacts:
            if stage == "separation":
                manifest.stems.update(artifacts)
            elif stage == "analysis":
                manifest.analysis = artifacts
            elif stage == "structure":
                manifest.timeline = artifacts.get("timeline", artifacts)
            elif stage == "evaluation":
                manifest.evaluation = artifacts
            elif stage == "transcription":
                manifest.midis.update(artifacts)
        if stage in manifest.stage_errors:
            del manifest.stage_errors[stage]
        self.save_manifest(file_path, manifest)
        return manifest

    def record_stage_error(
        self,
        file_path: Union[str, Path],
        stage: str,
        error_message: str,
    ) -> WorkspaceManifest:
        manifest = self.get_manifest(file_path)
        manifest.status = "failed"
        manifest.current_stage = stage
        manifest.stage_errors[stage] = error_message
        self.save_manifest(file_path, manifest)
        return manifest
