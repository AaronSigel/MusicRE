import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from music_re.core.analyzer import analyze_audio
from music_re.core.midi_digest import generate_midi_digest
from music_re.core.midi_drums import transcribe_drums
from music_re.core.midi_tonal import transcribe_tonal
from music_re.core.separator import separate_stems
from music_re.core.slicer import slice_audio
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


@app.command("split")
def cmd_split(
    file: Path = typer.Argument(..., help="Path to input audio file"),
    six_stems: bool = typer.Option(False, "--six-stems", "-6", help="Extract 6 stems (includes guitar & piano)"),
):
    """Separate audio into stems (Demucs) using cache."""
    ws = WorkspaceManager()
    console.print(f"[cyan]Separating stems for {file.name}...[/cyan]")
    result = separate_stems(file, workspace=ws, six_stems=six_stems, progress_callback=console.print)

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
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output MIDI path"),
):
    """Transcribe audio stem to MIDI."""
    if out is None:
        out = stem_file.with_suffix(".mid")

    if mode == "drums":
        res = transcribe_drums(stem_file, out, bpm=bpm)
    else:
        res = transcribe_tonal(stem_file, out, bpm=bpm)

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


@app.command("deconstruct")
def cmd_deconstruct(
    file: Path = typer.Argument(..., help="Path to input audio file"),
    six_stems: bool = typer.Option(False, "--six-stems", "-6", help="Extract 6 stems"),
):
    """Run full reverse engineering pipeline: split -> analyze -> midi transcribe."""
    ws = WorkspaceManager()
    console.print(f"[bold cyan]Starting deconstruct pipeline for {file.name}...[/bold cyan]")

    # 1. Split stems
    split_res = separate_stems(file, workspace=ws, six_stems=six_stems, progress_callback=console.print)

    # 2. Analyze tempo using drums or full track
    analysis_src = split_res.stems.get("drums", str(file))
    analysis = analyze_audio(analysis_src)
    bpm = analysis.bpm if analysis.bpm > 0 else 120.0
    ws.save_metadata(file, analysis.model_dump())
    console.print(f"[green]Detected BPM:[/green] {bpm:.2f}, [green]Key:[/green] {analysis.key}")

    # 3. Transcribe stems to MIDI
    midi_outputs = {}
    track_dir = ws.get_track_dir(file)
    midis_dir = track_dir / "midis"
    midis_dir.mkdir(parents=True, exist_ok=True)

    for stem_name, stem_path in split_res.stems.items():
        out_mid = midis_dir / f"{stem_name}.mid"
        if stem_name == "drums":
            transcribe_drums(stem_path, out_mid, bpm=bpm)
        else:
            transcribe_tonal(stem_path, out_mid, bpm=bpm)
        midi_outputs[stem_name] = str(out_mid)

    console.print(f"[bold green]Deconstruct complete! Workspace:[/bold green] {track_dir}")
    for name, path in midi_outputs.items():
        console.print(f"  • {name}: {path}")
