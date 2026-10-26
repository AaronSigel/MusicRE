from pathlib import Path
import shutil
import tempfile
from typing import Union
import zipfile

from music_re.workspace import WorkspaceManager


def export_bundle(
    workspace: WorkspaceManager,
    file_path: Union[str, Path],
    target_path: Union[str, Path],
    profile: str = "balanced",
    archive: bool = False,
) -> Path:
    """Export all reversed audio stems, MIDIs, timeline, manifest, and quality report into a clean bundle."""
    p = Path(file_path).resolve()
    out_target = Path(target_path).resolve()

    track_dir = workspace.get_track_dir(p)

    # Defensive path safety checks
    if out_target == p:
        raise ValueError("Target path cannot overwrite source input file")
    if out_target == track_dir or track_dir in out_target.parents:
        raise ValueError("Target path cannot be inside the track cache workspace")

    manifest = workspace.get_manifest(p)

    def _copy_contents(destination_dir: Path) -> None:
        destination_dir.mkdir(parents=True, exist_ok=True)

        # 1. Stems
        stems_dest = destination_dir / "stems"
        stems_dest.mkdir(parents=True, exist_ok=True)
        for stem_name, stem_path_str in manifest.stems.items():
            src_f = Path(stem_path_str)
            if src_f.is_file():
                shutil.copy2(str(src_f), str(stems_dest / f"{stem_name}.wav"))

        # 2. MIDIs
        midis_dest = destination_dir / "midis"
        midis_dest.mkdir(parents=True, exist_ok=True)
        for midi_name, midi_path_str in manifest.midis.items():
            src_m = Path(midi_path_str)
            if src_m.is_file():
                shutil.copy2(str(src_m), str(midis_dest / f"{src_m.name}"))

        # Copy raw unquantized midis unless running under 'fast' profile
        if profile != "fast":
            src_midis_dir = track_dir / "midis"
            if src_midis_dir.is_dir():
                for raw_m in src_midis_dir.glob("*_raw.mid"):
                    shutil.copy2(str(raw_m), str(midis_dest / raw_m.name))

        # 3. Timeline
        tl_path = workspace.get_timeline_path(p)
        if tl_path.is_file():
            shutil.copy2(str(tl_path), str(destination_dir / "timeline.json"))

        # 4. Manifest
        m_path = workspace.get_manifest_path(p)
        if m_path.is_file():
            shutil.copy2(str(m_path), str(destination_dir / "manifest.json"))

        # 5. Quality Report
        qr_path = workspace.get_quality_report_path(p)
        if qr_path.is_file():
            shutil.copy2(str(qr_path), str(destination_dir / "quality_report.json"))

    if archive:
        zip_path = out_target if out_target.suffix == ".zip" else out_target.with_suffix(".zip")
        if zip_path == p:
            raise ValueError("Target path cannot overwrite source input file")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            temp_bundle_dir = Path(td) / p.stem
            _copy_contents(temp_bundle_dir)

            with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
                for file_entry in temp_bundle_dir.rglob("*"):
                    if file_entry.is_file():
                        arcname = file_entry.relative_to(temp_bundle_dir)
                        zf.write(str(file_entry), str(arcname))
        return zip_path

    _copy_contents(out_target)
    return out_target
