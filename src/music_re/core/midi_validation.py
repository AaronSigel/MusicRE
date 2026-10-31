"""Per-note spectral evidence and MIDI validation."""
from __future__ import annotations

import dataclasses
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional

import numpy as np
import pretty_midi

from music_re.core.spectral import (SpectralConfig, StemSpectralFeatures,
                                    compute_stem_features, load_mono, midi_to_hz)
from music_re.schemas import NoteEvidence, StemValidationReport


SPECTRAL_ALGO_VERSION = 'spectral-v1'


@dataclass(frozen=True)
class RoleRule:
    filter_enabled: bool
    reject_support: float
    reject_persistence: float
    uncertain_support: float
    uncertain_persistence: float


_DEFAULT_ROLES = (
    ('bass', RoleRule(True, .10, .20, .10, .20)),
    ('guitar', RoleRule(True, .10, .20, .10, .20)),
    ('other', RoleRule(True, .06, .10, .10, .20)),
    ('piano', RoleRule(False, .20, .20, .45, .50)),
    ('vocals', RoleRule(False, .20, .20, .45, .50)),
)


@dataclass(frozen=True)
class SpectralValidatorConfig:
    spectral: SpectralConfig = SpectralConfig()
    roles: tuple[tuple[str, RoleRule], ...] | Mapping[str, RoleRule] = _DEFAULT_ROLES

    def __post_init__(self):
        entries = tuple(sorted(self.roles.items())) if isinstance(self.roles, Mapping) else tuple(self.roles)
        object.__setattr__(self, 'roles', entries)

    def rule_for(self, stem: str) -> Optional[RoleRule]:
        return dict(self.roles).get(stem)

    def to_dict(self) -> dict:
        return {'spectral': self.spectral.to_dict(), 'roles': {
            name: dataclasses.asdict(rule) for name, rule in sorted(self.roles)
        }}


class SpectralMidiValidator:
    def __init__(self, config: SpectralValidatorConfig = SpectralValidatorConfig()):
        self.config = config

    def validate(self, audio_path, midi_path, stem_name: str,
                 mode: Literal['audit', 'filter'], out_midi_path=None) -> StemValidationReport:
        if mode not in ('audit', 'filter'):
            raise ValueError("mode must be 'audit' or 'filter'")
        if mode == 'filter':
            if out_midi_path is None:
                raise ValueError('filter mode requires out_midi_path')
            if Path(out_midi_path).resolve() == Path(midi_path).resolve():
                raise ValueError('out_midi_path must differ from midi_path')
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        instruments = midi.instruments
        notes_flat = [(inst, note) for inst in instruments for note in inst.notes]
        y = load_mono(audio_path, self.config.spectral.sr)
        rule = self.config.rule_for(stem_name)
        if stem_name == 'drums':
            status, decision = 'skipped', 'skipped'
        elif rule is None:
            status, decision = 'unsupported', 'unsupported'
            warning = f"Unsupported stem '{stem_name}'; MIDI passed through."
            warnings.warn(warning, UserWarning)
        else:
            status, decision = 'validated', 'unscored'
        if mode == 'audit' or (rule is None or not rule.filter_enabled or stem_name == 'drums'):
            if mode == 'filter':
                shutil.copyfile(midi_path, out_midi_path)
            if mode == 'audit' and rule is not None:
                features = compute_stem_features(y, self.config.spectral)
                evidence = [compute_note_evidence(features, n.pitch, n.start, n.end, n.velocity)
                            for _, n in notes_flat]
                for item in evidence:
                    item.decision = 'unscored'
            else:
                evidence = [NoteEvidence(pitch=n.pitch, start_sec=n.start, end_sec=n.end,
                                         velocity=n.velocity, harmonic_support=0, spectral_contrast=0,
                                         temporal_persistence=0, onset_agreement=0, leakage=0,
                                         decision=decision) for _, n in notes_flat]
            return self._report(stem_name, mode, status, midi_path,
                                out_midi_path if mode == 'filter' else None, evidence,
                                [warning] if rule is None and stem_name != 'drums' else [])
        features = compute_stem_features(y, self.config.spectral)
        evidence = []
        rejected = set()
        for inst in instruments:
            for note in inst.notes:
                item = compute_note_evidence(features, note.pitch, note.start, note.end, note.velocity)
                if (item.harmonic_support < rule.reject_support and
                        item.temporal_persistence < rule.reject_persistence):
                    item.decision = 'reject'
                    rejected.add(id(note))
                elif (item.harmonic_support < rule.uncertain_support or
                      item.temporal_persistence < rule.uncertain_persistence):
                    item.decision = 'uncertain'
                else:
                    item.decision = 'keep'
                evidence.append(item)
        for inst in instruments:
            inst.notes = [note for note in inst.notes if id(note) not in rejected]
        midi.write(str(out_midi_path))
        return self._report(stem_name, mode, status, midi_path, out_midi_path, evidence, [])

    @staticmethod
    def _report(stem, mode, status, input_path, output_path, notes, warnings_list):
        counts = {}
        for note in notes:
            counts[note.decision] = counts.get(note.decision, 0) + 1
        return StemValidationReport(stem=stem, mode=mode, status=status,
                                    input_midi_path=str(input_path),
                                    output_midi_path=str(output_path) if output_path is not None else None,
                                    notes=notes, counts=counts, warnings=warnings_list)


def _bounded(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0


def compute_note_evidence(
    features: StemSpectralFeatures,
    pitch: int,
    start_sec: float,
    end_sec: float,
    velocity: int = 100,
) -> NoteEvidence:
    """Measure pitch evidence in a note window; scores are descriptive, not probabilities."""
    n = len(features.times)
    zero = dict(harmonic_support=0.0, spectral_contrast=0.0,
                temporal_persistence=0.0, onset_agreement=0.0, leakage=0.0)
    reason = None
    if end_sec <= start_sec:
        return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec,
                            velocity=velocity, reason='invalid_duration', **zero)
    if n == 0 or end_sec <= 0 or start_sec >= (features.times[-1] + features.hop_length / features.sr):
        reason = 'out_of_range'
        return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec,
                            velocity=velocity, reason=reason, **zero)
    hop = features.hop_length / features.sr
    lo = max(0, int(np.floor(start_sec / hop)))
    hi = min(n, int(np.ceil(end_sec / hop)))
    if hi <= lo:
        hi = min(n, lo + 1)
    if hi <= lo:
        return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec,
                            velocity=velocity, reason='out_of_range', **zero)
    frames = np.arange(lo, hi)
    cqt_idx = np.flatnonzero(np.isclose(features.cqt_midi, pitch, atol=0.51))
    cqt_in_range = bool(cqt_idx.size)
    if not cqt_in_range:
        reason = 'pitch_outside_cqt_range_stft_only'
    f0 = midi_to_hz(pitch)
    mag = features.stft_mag[:, frames]
    freqs = features.stft_freqs
    # Integrate narrow fundamental and harmonic bands.
    harmonic = np.zeros(len(frames), dtype=np.float64)
    fundamental = np.zeros(len(frames), dtype=np.float64)
    harmonic_presence = np.zeros(len(frames), dtype=np.float64)
    for k in range(1, 5):
        mask = np.abs(freqs - f0 * k) <= max(12.0, f0 * k * 0.018)
        if mask.any():
            band = np.max(mag[mask], axis=0)
            harmonic += band / k
            nearby = np.abs(freqs - f0 * k)
            background_mask = (nearby >= max(25.0, f0 * k * 0.03)) & (nearby <= max(80.0, f0 * k * 0.12))
            background_level = np.mean(mag[background_mask], axis=0) if background_mask.any() else np.zeros(len(frames))
            harmonic_presence += band / (band + 8.0 * background_level + 1e-12)
            if k == 1:
                fundamental = band
    harmonic_presence /= 4.0
    cqt = np.mean(features.cqt_mag[cqt_idx][:, frames], axis=0) if cqt_in_range else fundamental
    if np.max(features.rms[frames]) <= 1e-10:
        return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec,
                            velocity=velocity, reason=reason, **zero)
    if cqt_in_range:
        cache = getattr(features, '_note_validation_cache', None)
        if cache is None:
            cache = {}
            setattr(features, '_note_validation_cache', cache)
        if pitch not in cache:
            pitch_track = np.mean(features.cqt_mag[cqt_idx], axis=0)
            cache[pitch] = (pitch_track, float(np.max(pitch_track)) * 0.08)
        pitch_track, persistence_threshold = cache[pitch]
        persistence = float(np.mean(pitch_track[frames] > persistence_threshold))
    else:
        persistence_threshold = float(np.max(fundamental)) * .08
        persistence = float(np.mean(fundamental > persistence_threshold))
    window_rms = features.rms[frames] + 1e-12
    fundamental_relative = np.clip(cqt / (25.0 * window_rms), 0.0, 1.0)
    support_frames = (0.2 * fundamental_relative + 0.8 * np.clip(harmonic_presence, 0, 1)
                      if cqt_in_range else np.clip(harmonic_presence, 0, 1))
    support = float(np.mean(support_frames))
    background_levels = []
    for k in range(1, 5):
        target = f0 * k
        distance = np.abs(freqs - target)
        local = (freqs >= target * 2 ** (-3 / 12)) & (freqs <= target * 2 ** (3 / 12))
        harmonic_bin = distance <= max(12, target * .018)
        local &= ~harmonic_bin
        if local.any():
            background_levels.append(np.mean(mag[local], axis=0))
    background = np.mean(background_levels, axis=0) if background_levels else np.zeros(len(frames))
    contrast_energy = harmonic if cqt_in_range else fundamental
    contrast = np.mean(contrast_energy / (contrast_energy + background + 1e-12))
    neighbor_energy = []
    for p in (pitch - 12, pitch - 1, pitch + 1, pitch + 12):
        idx = np.flatnonzero(np.isclose(features.cqt_midi, p, atol=.51))
        neighbor_energy.append(float(np.mean(features.cqt_mag[idx][:, frames])) if idx.size else 0.0)
    expected = float(np.mean(cqt))
    leakage = max((v - expected) / (v + expected + 1e-12) for v in neighbor_energy)
    leakage = max(0.0, leakage)
    flux = features.flux
    onset_idx = int(np.clip(round(start_sec / hop), 0, len(flux) - 1))
    window = 3
    near = flux[max(0, onset_idx-window):min(len(flux), onset_idx+window+1)]
    elsewhere_max = 0.0
    left_end = min(hi, onset_idx - window)
    if lo < left_end:
        elsewhere_max = max(elsewhere_max, float(np.max(flux[lo:left_end])))
    right_start = max(lo, onset_idx + window + 1)
    if right_start < hi:
        elsewhere_max = max(elsewhere_max, float(np.max(flux[right_start:hi])))
    near_max = float(np.max(near)) if near.size else 0.0
    onset = near_max / (elsewhere_max + 1e-12) if elsewhere_max > 0 else (1.0 if near_max > 0 else 0.0)
    return NoteEvidence(pitch=pitch, start_sec=start_sec, end_sec=end_sec, velocity=velocity,
                        harmonic_support=_bounded(support), spectral_contrast=_bounded(contrast),
                        temporal_persistence=_bounded(persistence), onset_agreement=_bounded(onset),
                        leakage=_bounded(leakage), reason=reason)
