import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from music_re.schemas import MusicalTimeline, QualityReport, WorkspaceManifest


def active_midis(manifest: WorkspaceManifest) -> Dict[str, str]:
    return manifest.validated_midis if manifest.active_midi_variant == 'validated' else manifest.midis


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


def _atomic_write_json(path: Path, data: Any) -> Path:
    """Write JSON via a unique sibling temp file, fsync, then atomically replace."""
    path = Path(path)
    tmp = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    try:
        with tmp:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp.name)
        raise
    return path


# Per-thread nesting state: {lock_path: [pid, depth, fd]}.
_held_locks = threading.local()


class WorkspaceManager:
    """Manages cache workspace directories indexed by content hash."""

    def __init__(self, cache_root: Optional[Union[str, Path]] = None):
        if cache_root is None:
            self.cache_root = Path.home() / ".music_re_cache"
        else:
            self.cache_root = Path(cache_root).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def lock(self, file_path: Union[str, Path]) -> Iterator[None]:
        """Exclusive inter-process lock on the track workspace; re-entrant per thread."""
        lock_path = str(self.get_track_dir(file_path) / ".lock")
        held = getattr(_held_locks, "held", None)
        if held is None:
            held = _held_locks.held = {}
        entry = held.get(lock_path)
        if entry is not None and entry[0] == os.getpid():
            entry[1] += 1
            try:
                yield
            finally:
                entry[1] -= 1
            return
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            held[lock_path] = [os.getpid(), 1, fd]
            try:
                yield
            finally:
                del held[lock_path]
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

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
        return _atomic_write_json(meta_file, metadata)

    def get_timeline_path(self, file_path: Union[str, Path]) -> Path:
        return self.get_track_dir(file_path) / "timeline.json"

    def save_timeline(self, file_path: Union[str, Path], timeline: MusicalTimeline) -> Path:
        tl_path = self.get_timeline_path(file_path)
        with self.lock(file_path):
            _atomic_write_json(tl_path, timeline.model_dump())
            self.mark_stage_complete(file_path, "structure", artifacts={"timeline": timeline.model_dump()})
        return tl_path

    def load_timeline(self, file_path: Union[str, Path]) -> Optional[MusicalTimeline]:
        if 'structure' not in self.get_manifest(file_path).completed_stages:
            return None
        tl_path = self.get_timeline_path(file_path)
        if not tl_path.exists():
            return None
        with open(tl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MusicalTimeline(**data)

    def get_quality_report_path(self, file_path: Union[str, Path]) -> Path:
        return self.get_track_dir(file_path) / "quality_report.json"

    def get_validated_midis_dir(self, file_path: Union[str, Path]) -> Path:
        return self.get_track_dir(file_path) / 'midis_validated'

    def get_spectral_validation_path(self, file_path: Union[str, Path]) -> Path:
        return self.get_track_dir(file_path) / 'spectral_validation.json'

    def save_quality_report(self, file_path: Union[str, Path], report: QualityReport) -> Path:
        qr_path = self.get_quality_report_path(file_path)
        with self.lock(file_path):
            _atomic_write_json(qr_path, report.model_dump())
            manifest = self.mark_stage_complete(file_path, "evaluation", artifacts={"report": report.model_dump()})
            # Same shape as music_re.core.pipeline._hashes, so export validation
            # still passes after a standalone re-evaluation.
            manifest.artifact_hashes["evaluation"] = {str(qr_path): compute_file_content_hash(qr_path)}
            self.save_manifest(file_path, manifest)
        return qr_path

    def load_quality_report(self, file_path: Union[str, Path]) -> Optional[QualityReport]:
        if 'evaluation' not in self.get_manifest(file_path).completed_stages:
            return None
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

        with self.lock(file_path):
            # Another process may have created it while we waited.
            if m_path.exists():
                with open(m_path, "r", encoding="utf-8") as f:
                    return WorkspaceManifest(**json.load(f))
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
        return _atomic_write_json(self.get_manifest_path(file_path), manifest.model_dump())

    def update_manifest(self, file_path: Union[str, Path], **kwargs) -> WorkspaceManifest:
        with self.lock(file_path):
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
        with self.lock(file_path):
            manifest = self.get_manifest(file_path)
            if artifacts is not None:
                if stage == "separation":
                    if manifest.stems != artifacts:
                        manifest = self.invalidate_stages(file_path, ['analysis', 'structure', 'transcription', 'spectral_validation', 'evaluation'])
                    manifest.stems = dict(artifacts)
                elif stage == "analysis":
                    manifest.analysis = artifacts
                elif stage == "structure":
                    if manifest.timeline != artifacts.get('timeline', artifacts):
                        manifest = self.invalidate_stages(file_path, ['transcription', 'spectral_validation', 'evaluation'])
                    manifest.timeline = artifacts.get("timeline", artifacts)
                elif stage == "evaluation":
                    manifest.evaluation = artifacts
                elif stage == "transcription":
                    if manifest.midis != artifacts and 'spectral_validation' in manifest.completed_stages:
                        manifest = self.invalidate_stages(file_path, ['spectral_validation', 'evaluation'])
                    manifest.midis = dict(artifacts)
            if stage not in manifest.completed_stages:
                manifest.completed_stages.append(stage)
            if stage in manifest.stage_errors:
                del manifest.stage_errors[stage]
            self.save_manifest(file_path, manifest)
            return manifest

    def invalidate_stages(self, file_path: Union[str, Path], stages) -> WorkspaceManifest:
        """Invalidate active state without deleting user or historical artifacts."""
        with self.lock(file_path):
            manifest = self.get_manifest(file_path)
            stages = list(stages)
            if 'transcription' in stages and 'spectral_validation' not in stages:
                stages.extend(['spectral_validation', 'evaluation'])
            for stage in stages:
                manifest.completed_stages = [s for s in manifest.completed_stages if s != stage]
                manifest.stage_fingerprints.pop(stage, None)
                manifest.artifact_hashes.pop(stage, None)
                manifest.stage_errors.pop(stage, None)
                if stage == 'separation':
                    manifest.stems = {}
                elif stage == 'analysis':
                    manifest.analysis = None
                elif stage == 'structure':
                    manifest.timeline = None
                elif stage == 'transcription':
                    manifest.midis = {}
                    manifest.transcription_results = {}
                elif stage == 'evaluation':
                    manifest.evaluation = None
                elif stage == 'spectral_validation':
                    manifest.validated_midis = {}
                    manifest.active_midi_variant = 'original'
                    manifest.spectral_validation = None
            manifest.status = 'pending'
            self.save_manifest(file_path, manifest)
            return manifest

    def record_stage_error(
        self,
        file_path: Union[str, Path],
        stage: str,
        error_message: str,
    ) -> WorkspaceManifest:
        with self.lock(file_path):
            manifest = self.get_manifest(file_path)
            manifest.status = "failed"
            manifest.current_stage = stage
            manifest.stage_errors[stage] = error_message
            self.save_manifest(file_path, manifest)
            return manifest
