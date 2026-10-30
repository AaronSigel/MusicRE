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
from music_re.core.pipeline import run_pipeline
from music_re.core.quality import evaluate_quality
from music_re.core.separator import separate_stems
from music_re.core.slicer import slice_audio
from music_re.schemas import MusicalTimeline
from music_re.workspace import WorkspaceManager

app = typer.Typer(help='music-re: Audio Reverse Engineering CLI')
console = Console()


@app.command('slice')
def cmd_slice(file: Path, start: float = typer.Option(0, '--start', '-s'),
              end: Optional[float] = typer.Option(None, '--end', '-e'),
              out: Optional[Path] = typer.Option(None, '--out', '-o')):
    """Extract an audio segment."""
    start_str = f"{start:g}"
    end_str = f"{end:g}" if end is not None else "0"
    out = out or file.with_name(f'{file.stem}_slice_{start_str}_{end_str}{file.suffix}')
    console.print(f'Saved slice to: {slice_audio(file, out, start_sec=start, end_sec=end)}')


@app.command('analyze')
def cmd_analyze(file: Path, as_json: bool = typer.Option(False, '--json')):
    """Analyze tempo, key, loudness, and beats."""
    analysis = analyze_audio(file)
    if as_json:
        console.print(analysis.model_dump_json(indent=2))
        return
    table = Table(title=f'Acoustic Analysis: {file.name}')
    table.add_column('Property')
    table.add_column('Value')
    for name, value in [('Duration', f'{analysis.duration_sec:.2f} s'), ('BPM', f'{analysis.bpm:.2f}'),
                         ('Key', f'{analysis.key} (confidence: {analysis.key_confidence:.2f})'),
                         ('LUFS', f'{analysis.lufs:.2f} LUFS'), ('Beat Count', str(len(analysis.beat_times)))]:
        table.add_row(name, value)
    console.print(table)


@app.command('structure')
def cmd_structure(file: Path, backend: str = typer.Option('auto', '--backend', '-b'),
                  device: str = typer.Option('auto', '--device', '-d'),
                  as_json: bool = typer.Option(False, '--json')):
    """Extract timeline; Librosa reports unknown labels and no downbeats."""
    ws = WorkspaceManager()
    analyzer = get_structure_analyzer(name=backend)
    timeline = analyzer.analyze(file, stems=ws.get_manifest(file).stems, device=device)
    ws.save_timeline(file, timeline)
    if as_json:
        console.print(timeline.model_dump_json(indent=2))
        return
    table = Table(title=f'Structure Analysis: {file.name} ({analyzer.backend_name})')
    table.add_column('Property')
    table.add_column('Value')
    for name, value in [('BPM', timeline.bpm), ('Total Beats', len(timeline.beats)),
                         ('Downbeats (Bars)', len(timeline.downbeats)), ('Sections Count', len(timeline.sections))]:
        table.add_row(name, str(value))
    console.print(table)
    sections = Table(title='Musical Sections')
    for name in ['Label', 'Start', 'End']:
        sections.add_column(name)
    for section in timeline.sections:
        sections.add_row(section.label, str(section.start_sec), str(section.end_sec))
    console.print(sections)


@app.command('split')
def cmd_split(file: Path, six_stems: bool = typer.Option(False, '--six-stems', '-6'),
              device: str = typer.Option('auto', '--device', '-d')):
    """Separate stems using Demucs."""
    result = separate_stems(file, workspace=WorkspaceManager(), six_stems=six_stems,
                            device=device, progress_callback=console.print)
    console.print(result.model_dump_json(indent=2))


@app.command('to-midi')
def cmd_to_midi(stem_file: Path, mode: str = typer.Option('tonal', '--mode', '-m'),
                bpm: float = typer.Option(120, '--bpm', '-b'),
                timeline: Optional[Path] = typer.Option(None, '--timeline', '-t'),
                raw: bool = typer.Option(True, '--raw/--no-raw'),
                grid: int = typer.Option(4, '--grid', '-g', min=1),
                onsets_only: bool = typer.Option(False, '--onsets-only'),
                out: Optional[Path] = typer.Option(None, '--out', '-o')):
    """Transcribe audio to MIDI; missing model output is an error."""
    if out is not None and out.resolve() == stem_file.resolve():
        raise typer.BadParameter('Output MIDI file cannot overwrite the input audio file.')
    tl = MusicalTimeline.model_validate_json(timeline.read_text()) if timeline else None
    result = get_transcriber_backend(mode=mode).transcribe(
        stem_file, out or stem_file.with_suffix('.mid'), timeline=tl, bpm=bpm,
        save_raw=raw, grid_subdivisions=grid, quantize_ends=not onsets_only)
    console.print(f'Saved {mode} MIDI ({result.status}) to: {result.midi_path}')


@app.command('inspect-midi')
def cmd_inspect_midi(midi_file: Path, bars: str = typer.Option('1-8', '--bars'),
                     bpm: float = typer.Option(120, '--bpm'),
                     as_json: bool = typer.Option(False, '--json')):
    """Inspect notes by bar and beat."""
    try:
        start, end = (int(b) for b in bars.split('-'))
    except ValueError:
        raise typer.BadParameter('bars must have form 1-8')
    digest = generate_midi_digest(midi_file, bpm=bpm, start_bar=start, end_bar=end)
    if as_json:
        console.print(digest.model_dump_json(indent=2))
        return
    table = Table(title=f'MIDI Digest: {midi_file.name} (Bars {bars}, BPM {bpm})')
    for name in ['Bar', 'Beat', 'Note', 'Pitch', 'Length (beats)', 'Velocity']:
        table.add_column(name)
    for note in digest.notes:
        table.add_row(str(note.bar), str(note.beat_in_bar), note.note, str(note.pitch),
                      str(note.duration_beats), str(note.velocity))
    console.print(table)
    console.print(f'Total notes in range: {digest.total_notes}')


@app.command('evaluate')
def cmd_evaluate(file: Path, ground_truth_stems: Optional[Path] = typer.Option(None, '--gt-stems'),
                  ground_truth_midi: Optional[Path] = typer.Option(None, '--gt-midi'),
                  as_json: bool = typer.Option(False, '--json')):
    """Evaluate reference metrics or report diagnostics."""
    ws = WorkspaceManager()
    report = evaluate_quality(ws, file, ground_truth_stems, ground_truth_midi)
    ws.save_quality_report(file, report)
    if as_json:
        console.print(report.model_dump_json(indent=2))
        return
    table = Table(title=f'Quality Evaluation: {file.name}')
    table.add_column('Metric')
    table.add_column('Value')
    table.add_row('Reference comparisons available', str(report.is_objective))
    table.add_row('Separation mode', report.separation_metrics['mode'])
    table.add_row('Transcription mode', report.transcription_metrics['mode'])
    table.add_row('Peak Level (dBFS)', str(report.audio_diagnostics.get('peak_dbfs')))
    table.add_row('Average reference SDR (dB)', str(report.separation_metrics.get('average_sdr_db')))
    table.add_row('Pitched onset+pitch F1', str(report.transcription_metrics.get('pitched', {}).get('onset_pitch_f1')))
    table.add_row('Total Notes', str(report.transcription_metrics.get('total_notes')))
    console.print(table)


@app.command('export')
def cmd_export(file: Path, out: Path = typer.Option(..., '--out', '-o'),
                profile: str = typer.Option('balanced', '--profile', '-p'),
                archive: bool = typer.Option(False, '--zip', '-z'),
                cache_root: Optional[Path] = typer.Option(None, '--cache-root')):
    """Export a complete deliverable bundle."""
    result = export_bundle(WorkspaceManager(cache_root=cache_root), file, out, profile, archive)
    console.print(f'Export complete: {result}')


@app.command('deconstruct')
def cmd_deconstruct(file: Path, six_stems: bool = typer.Option(False, '--six-stems', '-6'),
                     device: str = typer.Option('auto', '--device', '-d'),
                     profile: str = typer.Option('balanced', '--profile', '-p'),
                     cache_root: Optional[Path] = typer.Option(None, '--cache-root'),
                     as_json: bool = typer.Option(False, '--json')):
    """Run the shared resumable pipeline."""
    try:
        result = run_pipeline(file, workspace=WorkspaceManager(cache_root=cache_root), six_stems=six_stems,
                              device=device, profile=profile, progress_callback=None if as_json else console.print)
    except Exception as e:
        Console(stderr=True).print(f'[red]Pipeline failed:[/red] {e}')
        raise typer.Exit(1)
    if as_json:
        typer.echo(json.dumps(result, allow_nan=False))
    else:
        console.print(f'Deconstruct complete! Workspace: {result["workspace_dir"]}')
