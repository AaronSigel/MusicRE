"""Real-model CLI/stdio-MCP acceptance. Never replaces adapters with fakes."""
import argparse
import asyncio
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from music_re.core.audio_io import decode_audio

REPO = Path(__file__).resolve().parents[1]


def run_command(command, log_path, timeout=3600):
    """Keep evidence on failure, and propagate the failing exit status."""
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    Path(log_path).write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    result.check_returncode()
    return result.stdout


def inspect_export(directory):
    root = Path(directory)
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Missing export manifest')
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'completed' or not manifest['stems'] or 'all_stems' not in manifest['midis']:
        raise ValueError('Incomplete export manifest')
    if set(manifest['midis']) != set(manifest['stems']) | {'all_stems'}:
        raise ValueError('MIDI/stem sets disagree')
    summary = dict(stems={}, midis={}, midi_events={}, config=manifest['run_config'], runtime=manifest['runtime'])
    for role, path in manifest['stems'].items():
        if Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('Nonportable stem path')
        info = sf.info(root / path)
        summary['stems'][role] = dict(frames=info.frames, sample_rate=info.samplerate,
                                      channels=info.channels, duration=info.duration)
    for role, path in manifest['midis'].items():
        if Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('Nonportable MIDI path')
        pm = pretty_midi.PrettyMIDI(str(root / path))
        notes = [n for i in pm.instruments for n in i.notes]
        if any(n.end <= n.start or n.start < 0 for n in notes):
            raise ValueError('Invalid MIDI note timing')
        summary['midis'][role] = dict(notes=len(notes), tracks=[i.name for i in pm.instruments])
        summary['midi_events'][role] = sorted((n.pitch, n.velocity, n.start, n.end) for n in notes)
        if role != 'all_stems':
            result = manifest['transcription_results'][role]
            if result['note_count'] != len(notes) or (result['status'] == 'no_notes') != (len(notes) == 0):
                raise ValueError('MIDI/result state mismatch')
            if manifest['run_config']['save_raw']:
                pretty_midi.PrettyMIDI(str(root / result['raw_midi_path']))
    timeline = json.loads((root / 'timeline.json').read_text())
    if any(b < a for a, b in zip(timeline['beats'], timeline['beats'][1:])):
        raise ValueError('Nonmonotonic beats')
    summary['timeline'] = dict(duration=timeline['duration_sec'], bpm=timeline['bpm'],
                                backend=timeline['backend'], model=timeline['model'],
                                beats=len(timeline['beats']), downbeats=len(timeline['downbeats']),
                                sections=len(timeline['sections']))
    summary['results'] = manifest['transcription_results']
    return summary


def _snapshot(workspace):
    return {str(p): p.stat().st_mtime_ns for p in Path(workspace).rglob('*')
            if p.is_file() and p.name != 'manifest.json'}


def tool_payload(result):
    """FastMCP wraps generic return annotations in structuredContent.result."""
    if result.isError:
        raise RuntimeError(str(result.content))
    if result.structuredContent is not None:
        payload = result.structuredContent
        return payload['result'] if set(payload) == {'result'} else payload
    return json.loads(result.content[0].text)


async def _mcp_runs(source, root, profile):
    params = StdioServerParameters(command=sys.executable, args=['-m', 'music_re.mcp_server'],
                                   cwd=str(REPO), env=dict(os.environ))
    with (root / 'mcp.log').open('w') as log:
        async with stdio_client(params, errlog=log) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                async def call(name, arguments):
                    result = await session.call_tool(name, arguments)
                    return tool_payload(result)
                args = dict(file_path=str(source), cache_root=str(root / 'mcp-cache'), profile=profile, device='cpu')
                started = time.perf_counter()
                result = await call('deconstruct_full_pipeline', args)
                elapsed = time.perf_counter() - started
                before = _snapshot(result['workspace_dir'])
                await call('deconstruct_full_pipeline', args)
                if before != _snapshot(result['workspace_dir']):
                    raise ValueError('MCP warm run rewrote cached artifacts')
                await call('export_project_bundle', dict(file_path=str(source), cache_root=args['cache_root'],
                            target_dir=str(root / 'mcp-bundle'), profile=profile))
                await call('export_project_bundle', dict(file_path=str(source), cache_root=args['cache_root'],
                            target_dir=str(root / 'mcp-bundle.zip'), profile=profile, archive=True))
                return elapsed


def verify(source, output_root, profile='balanced'):
    source, root = Path(source).resolve(), Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = dict(status='failed', started_at=datetime.now(timezone.utc).isoformat(),
                  source=str(source), profile=profile)
    try:
        with source.open('rb') as stream:
            report['source_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        decoded, sr = decode_audio(source, target_sr=44100, normalize_db=None)
        report['decoded_frames'] = len(decoded)
        report['duration_sec'] = len(decoded) / sr
        command = [str(REPO / '.venv/bin/music-re'), 'deconstruct', str(source), '--profile', profile,
                   '--device', 'cpu', '--cache-root', str(root / 'cli-cache'), '--json']
        started = time.perf_counter()
        result = json.loads(run_command(command, root / 'cli-cold.log'))
        report['cli_seconds'] = time.perf_counter() - started
        before = _snapshot(result['workspace_dir'])
        run_command(command, root / 'cli-warm.log')
        if before != _snapshot(result['workspace_dir']):
            raise ValueError('CLI warm run rewrote cached artifacts')
        for archive in (False, True):
            run_command([str(REPO / '.venv/bin/music-re'), 'export', str(source), '--profile', profile,
                         '--cache-root', str(root / 'cli-cache'), '--out', str(root / ('cli-bundle.zip' if archive else 'cli-bundle'))]
                        + (['--zip'] if archive else []), root / f'cli-export-{archive}.log')
        report['mcp_seconds'] = asyncio.run(_mcp_runs(source, root, profile))
        cli, mcp = inspect_export(root / 'cli-bundle'), inspect_export(root / 'mcp-bundle')
        for key in ('config', 'runtime', 'results', 'stems', 'midis', 'timeline'):
            if cli[key] != mcp[key]:
                raise ValueError(f'CLI/MCP parity mismatch: {key}')
        for role, events in cli['midi_events'].items():
            other = mcp['midi_events'][role]
            if len(events) != len(other) or any(a[:2] != b[:2] or abs(a[2]-b[2]) > .003 or abs(a[3]-b[3]) > .003
                                               for a, b in zip(events, other)):
                raise ValueError(f'CLI/MCP MIDI timing mismatch: {role}')
        if cli['midis']['all_stems']['notes'] != sum(v['notes'] for k, v in cli['midis'].items() if k != 'all_stems'):
            raise ValueError('Combined MIDI note count differs from active stem total')
        for stem in cli['stems'].values():
            if stem['frames'] != len(decoded) or stem['sample_rate'] != 44100:
                raise ValueError('Stem duration/sample rate differs from decoded input')
        if abs(cli['timeline']['duration'] - report['duration_sec']) > .001:
            raise ValueError('Timeline duration mismatch')
        for name in ['cli-bundle', 'mcp-bundle']:
            directory = root / name
            with zipfile.ZipFile(root / f'{name}.zip') as archive:
                expected = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
                if set(archive.namelist()) != expected:
                    raise ValueError('ZIP/directory artifact sets differ')
                for relative in expected:
                    if archive.read(relative) != (directory / relative).read_bytes():
                        raise ValueError('ZIP/directory contents differ')
        cli.pop('midi_events')
        report.update(status='passed', artifacts=cli, warm_resume='passed', parity='passed', midi_tolerance_sec=.003)
    except Exception as e:
        report['error'] = f'{type(e).__name__}: {e}'
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['peak_child_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        (root / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--profile', choices=['fast', 'balanced', 'quality'], default='balanced')
    args = parser.parse_args()
    print(json.dumps(verify(args.input, args.out, args.profile), indent=2))
