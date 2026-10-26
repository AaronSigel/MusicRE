import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from music_re.core.analyzer import analyze_audio
from music_re.core.backends import get_structure_analyzer, get_transcriber_backend
from music_re.core.export import export_bundle
from music_re.core.midi_digest import generate_midi_digest
from music_re.core.midi_drums import transcribe_drums
from music_re.core.midi_project import build_combined_midi_project
from music_re.core.midi_tonal import transcribe_tonal
from music_re.core.quality import evaluate_quality
from music_re.core.separator import separate_stems
from music_re.core.slicer import slice_audio
from music_re.schemas import MusicalTimeline
from music_re.workspace import WorkspaceManager

app = typer.Typer(help="music-re: Audio Reverse Engineering CLI")
console = Console()


@app.command("slice")
def cmd_slice(
    file: Path = typer.Argument(..., help="Path to input audio file"),
    start: float = typer.Option(0.0, "--start", "-s", help="Start time in seconds"),
    end: Optional[float] = typer.Option(None, "--end", "-e", help="End time in seconds"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output path for sliced audio"),
):
    """Extract audio slice by start and end timestamps."""
    if out is None:
        out = file.with_name(f"{file.stem}_slice_{int(start)}_{int(end or 0)}{file.suffix}")

    res = slice_audio(file, out, start_sec=start, end_sec=end)
    console.print(f"[green]Saved slice to:[/green] {res}")


@app.command("analyze")
def cmd_analyze(
    file: Path = typer.Argument(..., help="Path to input audio file"),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Analyze tempo, key, loudness (LUFS), and beat grid."""
    analysis = analyze_audio(file)

    if as_json:
        console.print(analysis.model_dump_json(indent=2))
        return

    table = Table(title=f"Acoustic Analysis: {file.name}")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Duration", f"{analysis.duration_sec:.2f} s")
    table.add_row("BPM", f"{analysis.bpm:.2f}")
    table.add_row("Key", f"{analysis.key} (confidence: {analysis.key_confidence:.2f})")
    table.add_row("LUFS", f"{analysis.lufs:.2f} LUFS")
    table.add_row("Beat Count", str(len(analysis.beat_times)))

    console.print(table)


@app.command("structure")
def cmd_structure(
    file: Path = typer.Argument(..., help="Path to input audio file"),
    backend: str = typer.Option("auto", "--backend", "-b", help="Structure backend: 'auto', 'allin1', 'librosa'"),
    device: str = typer.Option("auto", "--device", "-d", help="Computation device: 'auto', 'cpu', 'cuda'"),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Analyze musical structure: BPM, beats, downbeats, and sections (timeline.json)."""
    ws = WorkspaceManager()
    manifest = ws.get_manifest(file)
    analyzer = get_structure_analyzer(name=backend)
    console.print(f"[cyan]Analyzing structure for {file.name} using '{analyzer.backend_name}'...[/cyan]")
    timeline = analyzer.analyze(file, stems=manifest.stems, device=device)
    ws.save_timeline(file, timeline)

    if as_json:
        console.print(timeline.model_dump_json(indent=2))
        return

    table = Table(title=f"Structure Analysis: {file.name} ({analyzer.backend_name})")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Duration", f"{timeline.duration_sec:.2f} s")
    table.add_row("BPM", f"{timeline.bpm:.2f}")
    table.add_row("Total Beats", str(len(timeline.beats)))
    table.add_row("Downbeats (Bars)", str(len(timeline.downbeats)))
    table.add_row("Sections Count", str(len(timeline.sections)))
    console.print(table)

    if timeline.sections:
        sec_table = Table(title="Musical Sections")
        sec_table.add_column("Label", style="green")
        sec_table.add_column("Start", justify="right")
        sec_table.add_column("End", justify="right")
        sec_table.add_column("Duration", justify="right")
        for s in timeline.sections:
            sec_table.add_row(
                s.label,
                f"{s.start_sec:.2f}s",
                f"{s.end_sec:.2f}s",
                f"{s.end_sec - s.start_sec:.2f}s",
            )
        console.print(sec_table)


@app.command("split")
def cmd_split(
    file: Path = typer.Argument(..., help="Path to input audio file"),
    six_stems: bool = typer.Option(False, "--six-stems", "-6", help="Extract 6 stems (includes guitar & piano)"),
    device: str = typer.Option("auto", "--device", "-d", help="Computation device: 'auto', 'cpu', 'cuda'"),
):
    """Separate audio into stems (Demucs) using cache."""
    ws = WorkspaceManager()
    console.print(f"[cyan]Separating stems for {file.name}...[/cyan]")
    result = separate_stems(
        file,
        workspace=ws,
        six_stems=six_stems,
        device=device,
        progress_callback=console.print,
    )

    table = Table(title=f"Stems: {file.name} ({result.model_used})")
    table.add_column("Stem", style="cyan")
    table.add_column("Path", style="green")

    for stem_name, stem_path in result.stems.items():
        table.add_row(stem_name, stem_path)

    console.print(table)


@app.command("to-midi")
def cmd_to_midi(
    stem_file: Path = typer.Argument(..., help="Path to audio stem file"),
    mode: str = typer.Option("tonal", "--mode", "-m", help="Transcription mode: 'tonal' or 'drums'"),
    bpm: float = typer.Option(120.0, "--bpm", "-b", help="Tempo in BPM"),
    timeline: Optional[Path] = typer.Option(None, "--timeline", "-t", help="Path to timeline.json for dynamic snapping"),
    raw: bool = typer.Option(True, "--raw/--no-raw", help="Save unquantized raw copy"),
    grid: int = typer.Option(4, "--grid", "-g", help="Subdivisions per beat (e.g. 4 for 16th notes)"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output MIDI path"),
):
    """Transcribe audio stem to MIDI."""
    if out is None:
        out = stem_file.with_suffix(".mid")

    tl_obj = None
    if timeline and timeline.is_file():
        try:
            with open(timeline, "r", encoding="utf-8") as f:
                tl_obj = MusicalTimeline(**json.load(f))
        except Exception as e:
            console.print(f"[yellow]Warning: Could not parse timeline file {timeline}: {e}[/yellow]")

    transcriber = get_transcriber_backend(mode=mode)
    res = transcriber.transcribe(
        audio_path=stem_file,
        output_midi_path=out,
        timeline=tl_obj,
        bpm=bpm,
        quantize=True,
        save_raw=raw,
    )

    console.print(f"[green]Saved {mode} MIDI to:[/green] {res}")


@app.command("inspect-midi")
def cmd_inspect_midi(
    midi_file: Path = typer.Argument(..., help="Path to MIDI file"),
    bars: str = typer.Option("1-8", "--bars", help="Bar range, e.g. 1-8"),
    bpm: float = typer.Option(120.0, "--bpm", help="Tempo in BPM"),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Inspect MIDI notes structured by bars and beats for LLM context."""
    try:
        start_bar, end_bar = (int(b) for b in bars.split("-"))
    except Exception:
        start_bar, end_bar = 1, 8

    digest = generate_midi_digest(midi_file, bpm=bpm, start_bar=start_bar, end_bar=end_bar)

    if as_json:
        console.print(digest.model_dump_json(indent=2))
        return

    table = Table(title=f"MIDI Digest: {midi_file.name} (Bars {bars}, BPM {bpm})")
    table.add_column("Bar", justify="right", style="cyan")
    table.add_column("Beat", justify="right", style="green")
    table.add_column("Note", style="magenta")
    table.add_column("Pitch", justify="right")
    table.add_column("Length (beats)", justify="right")
    table.add_column("Velocity", justify="right")

    for n in digest.notes:
        table.add_row(
            str(n.bar),
            f"{n.beat_in_bar:.2f}",
            n.note,
            str(n.pitch),
            f"{n.duration_beats:.2f}",
            str(n.velocity),
        )

    console.print(table)
    console.print(f"[dim]Total notes in range: {digest.total_notes}[/dim]")


@app.command("evaluate")
def cmd_evaluate(
    file: Path = typer.Argument(..., help="Path to input audio file"),
    ground_truth_stems: Optional[Path] = typer.Option(None, "--gt-stems", help="Directory with ground truth stem WAV files"),
    ground_truth_midi: Optional[Path] = typer.Option(None, "--gt-midi", help="Ground truth MIDI file"),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Evaluate decomposition and transcription quality (quality_report.json)."""
    ws = WorkspaceManager()
    report = evaluate_quality(
        workspace=ws,
        file_path=file,
        ground_truth_stems=ground_truth_stems,
        ground_truth_midi=ground_truth_midi,
    )
    ws.save_quality_report(file, report)

    if as_json:
        console.print(report.model_dump_json(indent=2))
        return

    table = Table(title=f"Quality Evaluation: {file.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    mode_desc = "Objective (Ground Truth)" if report.is_objective else "Diagnostic Heuristic (No Ground Truth)"
    table.add_row("Evaluation Mode", mode_desc)
    table.add_row("Peak Level", f"{report.audio_diagnostics.get('peak_dbfs', 0.0):.2f} dBFS")
    table.add_row("Clipping Samples", str(report.audio_diagnostics.get('clipping_samples', 0)))

    if report.is_objective:
        table.add_row("Average SDR", f"{report.separation_metrics.get('average_sdr_db', 0.0):.2f} dB")
        if "onset_f1" in report.transcription_metrics:
            table.add_row("Note Onset F1", f"{report.transcription_metrics['onset_f1']:.3f}")
    else:
        table.add_row("Residual Reconstruction", f"{report.separation_metrics.get('residual_reconstruction_db', 0.0):.2f} dB")
        table.add_row("Total Notes", str(report.transcription_metrics.get('total_notes', 0)))

    console.print(table)


@app.command("export")
def cmd_export(
    file: Path = typer.Argument(..., help="Path to input audio file"),
    out: Path = typer.Option(..., "--out", "-o", help="Target output directory or zip file"),
    profile: str = typer.Option("balanced", "--profile", "-p", help="Export profile: 'fast', 'balanced', 'quality'"),
    archive: bool = typer.Option(False, "--zip", "-z", help="Export as a zip archive"),
):
    """Export all reversed audio stems, MIDIs, timeline, and reports into a clean deliverable bundle."""
    ws = WorkspaceManager()
    res = export_bundle(workspace=ws, file_path=file, target_path=out, profile=profile, archive=archive)
    console.print(f"[bold green]Export complete:[/bold green] {res}")


@app.command("deconstruct")
def cmd_deconstruct(
    file: Path = typer.Argument(..., help="Path to input audio file"),
    six_stems: bool = typer.Option(False, "--six-stems", "-6", help="Extract 6 stems"),
    device: str = typer.Option("auto", "--device", "-d", help="Computation device: 'auto', 'cpu', 'cuda'"),
    profile: str = typer.Option("balanced", "--profile", "-p", help="Execution profile: 'fast', 'balanced', 'quality'"),
):
    """Run full reverse engineering pipeline: split -> analyze -> midi transcribe with resumption and error handling."""
    ws = WorkspaceManager()
    track_dir = ws.get_track_dir(file)
    manifest = ws.get_manifest(file)
    console.print(f"[bold cyan]Starting deconstruct pipeline for {file.name}...[/bold cyan]")
    ws.update_manifest(file, status="processing")

    # Profile overrides
    if profile == "fast":
        six_stems = False

    # 1. Split stems
    expected_stem_count = 6 if six_stems else 4
    can_resume_separation = (
        "separation" in manifest.completed_stages
        and len(manifest.stems) == expected_stem_count
        and all(Path(p).is_file() for p in manifest.stems.values())
    )
    if can_resume_separation:
        console.print("[green]Reusing previously separated stems from workspace.[/green]")
        stems_dict = manifest.stems
    else:
        try:
            ws.update_manifest(file, current_stage="separation")
            split_res = separate_stems(
                file,
                workspace=ws,
                six_stems=six_stems,
                device=device,
                progress_callback=console.print,
            )
            stems_dict = split_res.stems
            ws.mark_stage_complete(file, "separation", artifacts=stems_dict)
        except Exception as e:
            ws.record_stage_error(file, "separation", str(e))
            console.print(f"[red]Separation stage failed:[/red] {e}")
            raise typer.Exit(code=1)

    # 2. Acoustic Analysis
    if "analysis" in manifest.completed_stages and manifest.analysis:
        console.print("[green]Reusing acoustic analysis from workspace.[/green]")
        bpm = manifest.analysis.get("bpm", 120.0)
        key = manifest.analysis.get("key", "N/A")
    else:
        try:
            ws.update_manifest(file, current_stage="analysis")
            # Analyze whole track for key and loudness, drums for tempo grid
            mix_analysis = analyze_audio(file)
            if "drums" in stems_dict:
                drum_analysis = analyze_audio(stems_dict["drums"], stem_name="drums")
                bpm = drum_analysis.bpm if drum_analysis.bpm > 0 else mix_analysis.bpm
            else:
                bpm = mix_analysis.bpm

            key = mix_analysis.key
            analysis_dict = {
                "bpm": bpm,
                "key": key,
                "key_confidence": mix_analysis.key_confidence,
                "lufs": mix_analysis.lufs,
                "duration_sec": mix_analysis.duration_sec,
                "beat_times": mix_analysis.beat_times,
            }
            ws.save_metadata(file, analysis_dict)
            ws.mark_stage_complete(file, "analysis", artifacts=analysis_dict)
            console.print(f"[green]Detected BPM:[/green] {bpm:.2f}, [green]Key:[/green] {key}")
        except Exception as e:
            ws.record_stage_error(file, "analysis", str(e))
            console.print(f"[red]Analysis stage failed:[/red] {e}")
            raise typer.Exit(code=1)

    # 3. Musical Structure Analysis (timeline.json)
    if "structure" in manifest.completed_stages and manifest.timeline:
        console.print("[green]Reusing musical structure (timeline.json) from workspace.[/green]")
        timeline = ws.load_timeline(file)
    else:
        try:
            ws.update_manifest(file, current_stage="structure")
            structure_analyzer = get_structure_analyzer(name="auto")
            timeline = structure_analyzer.analyze(file, stems=stems_dict, device=device)
            ws.save_timeline(file, timeline)
            console.print(
                f"[green]Structure analyzed:[/green] {len(timeline.sections)} sections, "
                f"{len(timeline.beats)} beats, {len(timeline.downbeats)} bars"
            )
            if timeline.bpm > 0:
                bpm = timeline.bpm
            if bpm <= 0:
                bpm = 120.0
        except Exception as e:
            ws.record_stage_error(file, "structure", str(e))
            console.print(f"[red]Structure analysis stage failed:[/red] {e}")
            raise typer.Exit(code=1)

    # 4. Transcribe stems to MIDI
    midis_dir = track_dir / "midis"
    midis_dir.mkdir(parents=True, exist_ok=True)
    midi_outputs = dict(manifest.midis)

    try:
        ws.update_manifest(file, current_stage="transcription")
        for stem_name, stem_path in stems_dict.items():
            if stem_name in midi_outputs and Path(midi_outputs[stem_name]).is_file():
                continue
            out_mid = midis_dir / f"{stem_name}.mid"
            mode = "drums" if stem_name == "drums" else "tonal"
            transcriber = get_transcriber_backend(mode=mode)
            transcriber.transcribe(
                audio_path=stem_path,
                output_midi_path=out_mid,
                timeline=timeline,
                bpm=bpm,
                quantize=True,
                save_raw=True,
            )
            midi_outputs[stem_name] = str(out_mid)

        # Assemble unified multi-track Standard MIDI project (all_stems.mid)
        all_stems_path = midis_dir / "all_stems.mid"
        build_combined_midi_project(
            stem_midis={k: v for k, v in midi_outputs.items() if k != "all_stems"},
            output_path=all_stems_path,
            timeline=timeline,
            bpm=bpm,
        )
        midi_outputs["all_stems"] = str(all_stems_path)

        ws.mark_stage_complete(file, "transcription", artifacts=midi_outputs)
        ws.update_manifest(file, status="completed", current_stage="completed")

        # 5. Evaluate quality if profile is balanced or quality
        if profile in ("balanced", "quality"):
            report = evaluate_quality(ws, file)
            ws.save_quality_report(file, report)
    except Exception as e:
        ws.record_stage_error(file, "transcription", str(e))
        console.print(f"[red]Transcription stage failed:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Deconstruct complete! Workspace:[/bold green] {track_dir}")
    for name, path in midi_outputs.items():
        console.print(f"  • {name}: {path}")
