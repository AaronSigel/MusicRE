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

from music_re.core.spectral import (SpectralConfig, StemSpectralFeatures, compute_rms,
                                    compute_stem_features, load_mono, midi_to_hz)
from music_re.schemas import NoteEvidence, StemValidationReport


SPECTRAL_ALGO_VERSION = 'spectral-v4'


# Order of the inputs to the calibrated note score (scripts/calibrate_spectral.py).
SCORE_FEATURES = ('harmonic_support', 'spectral_contrast', 'temporal_persistence',
                  'onset_agreement', 'leakage', 'log_duration')
_MIN_DURATION_SEC = 1e-3
# Notes without measurable evidence: the score is reported but never rejects them on its own.
_NO_EVIDENCE_REASONS = frozenset({'out_of_range', 'invalid_duration'})
_SCORE_FIELDS = ('score_weights', 'score_bias', 'score_mean', 'score_scale', 'reject_below', 'uncertain_below')


@dataclass(frozen=True)
class RoleRule:
    filter_enabled: bool
    reject_support: float
    reject_persistence: float
    uncertain_support: float
    uncertain_persistence: float
    # Optional calibrated logistic score; when set, filter decisions use it instead of the thresholds above.
    score_weights: tuple[float, ...] | None = None
    score_bias: float | None = None
    score_mean: tuple[float, ...] | None = None
    score_scale: tuple[float, ...] | None = None
    reject_below: float | None = None
    uncertain_below: float | None = None

    def __post_init__(self):
        values = [getattr(self, name) for name in _SCORE_FIELDS]
        if all(v is None for v in values):
            return
        if any(v is None for v in values):
            raise ValueError(f'score parameters must be all set or all None: {_SCORE_FIELDS}')
        for name in ('score_weights', 'score_mean', 'score_scale'):
            vector = tuple(float(v) for v in getattr(self, name))
            if len(vector) != len(SCORE_FEATURES) or not all(np.isfinite(vector)):
                raise ValueError(f'{name} needs {len(SCORE_FEATURES)} finite values')
            object.__setattr__(self, name, vector)
        if any(v <= 0 for v in self.score_scale):
            raise ValueError('score_scale values must be positive')
        for name in ('score_bias', 'reject_below', 'uncertain_below'):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, value)
        if not 0.0 <= self.reject_below <= self.uncertain_below <= 1.0:
            raise ValueError('need 0 <= reject_below <= uncertain_below <= 1')

    @property
    def has_score(self) -> bool:
        return self.score_weights is not None

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        if not self.has_score:
            return {k: v for k, v in data.items() if k not in _SCORE_FIELDS}
        return {k: list(v) if isinstance(v, tuple) else v for k, v in data.items()}


def score_features(item: NoteEvidence) -> np.ndarray:
    duration = max(item.end_sec - item.start_sec, _MIN_DURATION_SEC)
    return np.array([item.harmonic_support, item.spectral_contrast, item.temporal_persistence,
                     item.onset_agreement, item.leakage, np.log(duration)], dtype=np.float64)


def note_score(rule: RoleRule, item: NoteEvidence) -> Optional[float]:
    """Calibrated probability-like score in [0, 1]; None when the role has no score parameters."""
    if not rule.has_score:
        return None
    x = np.nan_to_num(score_features(item), nan=0.0, posinf=0.0, neginf=0.0)
    z = (x - np.asarray(rule.score_mean)) / np.asarray(rule.score_scale)
    logit = float(np.clip(np.dot(rule.score_weights, z) + rule.score_bias, -500.0, 500.0))
    return float(1.0 / (1.0 + np.exp(-logit)))


def score_decision(rule: RoleRule, score: float, reason: Optional[str]) -> str:
    if reason in _NO_EVIDENCE_REASONS:
        return 'uncertain'
    if score < rule.reject_below:
        return 'reject'
    if score < rule.uncertain_below:
        return 'uncertain'
    return 'keep'


_DEFAULT_ROLES = (
    ('bass', RoleRule(True, .10, .20, .10, .20)),
    ('guitar', RoleRule(True, .10, .20, .10, .20)),
    # Fitted by scripts/calibrate_spectral.py on the Slakh tune split (benchmark cache
    # 20261031T122948223612, octave-corrected references). Bass keeps the threshold rule:
    # its calibrated score lost 11% of test TPs (cap 2%).
    ('other', RoleRule(
        True, .06, .10, .10, .20,
        score_weights=(-0.1073164721798026, 0.18584598420812792, 0.09455851036997381,
                       0.5468716080373939, -0.1891921463719693, -0.05920147844021798),
        score_bias=0.024125373814499373,
        score_mean=(0.4846532206893725, 0.9494680549073412, 0.8900147971981125,
                    0.8115185129425667, 0.07157978047790219, -0.937571259628061),
        score_scale=(0.10863819537638837, 0.029523579900300453, 0.20134445264277767,
                     0.2537749746799757, 0.15306162355760436, 0.641796602897982),
        reject_below=0.23139222965527687, uncertain_below=0.3656961148276384)),
    ('piano', RoleRule(False, .20, .20, .45, .50)),
    ('vocals', RoleRule(False, .20, .20, .45, .50)),
)


# Roles whose separated stem can be judged absent; drums never are.
PRESENCE_ROLES = ('bass', 'guitar', 'other', 'piano', 'vocals')
_MIX_ACTIVE_DBFS = -60.0
_STEM_ACTIVE_DBFS = -50.0
_DB_FLOOR = -120.0


@dataclass(frozen=True)
class PresenceRule:
    """A stem is absent iff rel_db < min_rel_db AND median_support < min_median_support."""
    min_rel_db: float
    min_median_support: float

    def __post_init__(self):
        for name in ('min_rel_db', 'min_median_support'):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# Fitted by `scripts/calibrate_spectral.py presence` (BabySlakh cache 20261031T122948223612 plus
# 50 MUSDB18 7-s previews, profile fast). Vocals: the tune optimum min_rel_db -13.2 dB marked one
# test stem with real vocals absent (PR - Oh No, -27.7 dB). The -30 dB cap was chosen AFTER
# inspecting that test-split result (ledger ruling R4), so the test split is not a holdout for
# presence: its zero false-absent is an all-tracks figure, not an unbiased estimate. The cap
# only moves the rule in the conservative direction (fewer drops).
# Only calibrated roles get a rule. The other PRESENCE_ROLES (bass, other, guitar, piano) have no
# usable absent examples (bass: Track00017 is not separable from quiet present bass), so their
# verdict stays None (not evaluated) rather than an unfounded 'present'.
_DEFAULT_PRESENCE = (('vocals', PresenceRule(-30.0, 0.3759)),)


def _as_entries(value):
    return tuple(sorted(value.items())) if isinstance(value, Mapping) else tuple(value)


@dataclass(frozen=True)
class SpectralValidatorConfig:
    spectral: SpectralConfig = SpectralConfig()
    roles: tuple[tuple[str, RoleRule], ...] | Mapping[str, RoleRule] = _DEFAULT_ROLES
    presence: tuple[tuple[str, PresenceRule], ...] | Mapping[str, PresenceRule] = _DEFAULT_PRESENCE

    def __post_init__(self):
        object.__setattr__(self, 'roles', _as_entries(self.roles))
        object.__setattr__(self, 'presence', _as_entries(self.presence))

    def rule_for(self, stem: str) -> Optional[RoleRule]:
        return dict(self.roles).get(stem)

    def presence_for(self, stem: str) -> Optional[PresenceRule]:
        return None if stem == 'drums' else dict(self.presence).get(stem)

    def to_dict(self) -> dict:
        return {'spectral': self.spectral.to_dict(),
                'roles': {name: rule.to_dict() for name, rule in sorted(self.roles)},
                'presence': {name: rule.to_dict() for name, rule in sorted(self.presence)}}


def _frame_db(y: np.ndarray, config: SpectralConfig) -> np.ndarray:
    if len(y) == 0:
        return np.zeros(0)
    rms = compute_rms(y, frame_length=config.n_fft, hop_length=config.hop_length).astype(np.float64)
    return 20.0 * np.log10(np.maximum(rms, 10 ** (_DB_FLOOR / 20)))


def _active_db(frame_db: np.ndarray) -> float:
    """RMS level (dBFS) over the given frames."""
    if frame_db.size == 0:
        return _DB_FLOOR
    power = float(np.mean(10.0 ** (frame_db / 10.0)))
    return max(_DB_FLOOR, 10.0 * np.log10(power)) if power > 0 else _DB_FLOOR


def stem_presence(stem: np.ndarray, mix: Optional[np.ndarray], evidence, config: SpectralConfig,
                  rule: Optional[PresenceRule]) -> dict:
    """Presence features of a separated stem relative to its source mix, plus the verdict.

    rel_db: stem RMS minus mix RMS over the frames where the mix exceeds -60 dBFS.
    active_fraction: fraction of stem frames above -50 dBFS.
    median_support: median harmonic_support of the stem's notes (0 without notes).
    The verdict is None when there is no rule, no mix, or the mix is silent.
    """
    stem_db = _frame_db(stem, config)
    supports = [item.harmonic_support for item in evidence]
    presence = dict(rel_db=None,
                    active_fraction=float(np.mean(stem_db > _STEM_ACTIVE_DBFS)) if stem_db.size else 0.0,
                    median_support=float(np.median(supports)) if supports else 0.0,
                    verdict=None)
    if mix is None:
        return presence
    mix_db = _frame_db(mix, config)
    n = min(len(mix_db), len(stem_db))
    active = mix_db[:n] > _MIX_ACTIVE_DBFS
    if not active.any():
        return presence
    presence['rel_db'] = float(_active_db(stem_db[:n][active]) - _active_db(mix_db[:n][active]))
    if rule is not None:
        absent = presence['rel_db'] < rule.min_rel_db and presence['median_support'] < rule.min_median_support
        presence['verdict'] = 'absent' if absent else 'present'
    return presence


class SpectralMidiValidator:
    def __init__(self, config: SpectralValidatorConfig = SpectralValidatorConfig()):
        self.config = config
        self._mix_cache = None  # (key, samples) of the last decoded source mix

    def _load_mix(self, mix_audio_path) -> Optional[np.ndarray]:
        """Decode the source mix once per validator; every stem of a track shares it."""
        if mix_audio_path is None:
            return None
        path = Path(mix_audio_path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size, self.config.spectral.sr)
        if self._mix_cache is None or self._mix_cache[0] != key:
            self._mix_cache = (key, load_mono(mix_audio_path, self.config.spectral.sr))
        return self._mix_cache[1]

    def validate(self, audio_path, midi_path, stem_name: str,
                 mode: Literal['audit', 'filter'], out_midi_path=None,
                 mix_audio_path=None) -> StemValidationReport:
        if mode not in ('audit', 'filter'):
            raise ValueError("mode must be 'audit' or 'filter'")
        if mode == 'filter':
            if out_midi_path is None:
                raise ValueError('filter mode requires out_midi_path')
            if Path(out_midi_path).resolve() == Path(midi_path).resolve():
                raise ValueError('out_midi_path must differ from midi_path')
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        instruments = midi.instruments
        notes_flat = [note for inst in instruments for note in inst.notes]
        output = out_midi_path if mode == 'filter' else None
        rule = self.config.rule_for(stem_name)
        if stem_name == 'drums' or rule is None:
            if stem_name == 'drums':
                status, decision, warnings_list = 'skipped', 'skipped', []
            else:
                status, decision = 'unsupported', 'unsupported'
                warnings_list = [f"Unsupported stem '{stem_name}'; MIDI passed through."]
                warnings.warn(warnings_list[0], UserWarning)
            if mode == 'filter':
                shutil.copyfile(midi_path, out_midi_path)
            evidence = [NoteEvidence(pitch=n.pitch, start_sec=n.start, end_sec=n.end,
                                     velocity=n.velocity, harmonic_support=0, spectral_contrast=0,
                                     temporal_persistence=0, onset_agreement=0, leakage=0,
                                     decision=decision) for n in notes_flat]
            return self._report(stem_name, mode, status, midi_path, output, evidence, warnings_list)
        spectral = self.config.spectral
        y = load_mono(audio_path, spectral.sr)
        features = compute_stem_features(y, spectral)
        evidence = [compute_note_evidence(features, n.pitch, n.start, n.end, n.velocity) for n in notes_flat]
        for item in evidence:
            item.note_score = note_score(rule, item)
        mix = self._load_mix(mix_audio_path)
        presence = stem_presence(y, mix, evidence, spectral, self.config.presence_for(stem_name))
        absent = presence['verdict'] == 'absent'
        if mode == 'audit' or not (rule.filter_enabled or absent):
            for item in evidence:
                item.decision = 'unscored'
            if mode == 'filter':
                shutil.copyfile(midi_path, out_midi_path)
            return self._report(stem_name, mode, 'validated', midi_path, output, evidence, [], presence)
        for item in evidence:
            if absent:
                item.decision, item.reason = 'reject', 'stem_absent'
            elif rule.has_score:
                item.decision = score_decision(rule, item.note_score, item.reason)
            elif (item.harmonic_support < rule.reject_support and
                    item.temporal_persistence < rule.reject_persistence):
                item.decision = 'reject'
            elif (item.harmonic_support < rule.uncertain_support or
                  item.temporal_persistence < rule.uncertain_persistence):
                item.decision = 'uncertain'
            else:
                item.decision = 'keep'
        rejected = {id(note) for note, item in zip(notes_flat, evidence) if item.decision == 'reject'}
        for inst in instruments:
            inst.notes = [note for note in inst.notes if id(note) not in rejected]
        midi.write(str(out_midi_path))
        return self._report(stem_name, mode, 'absent' if absent else 'validated', midi_path,
                            out_midi_path, evidence, [], presence)

    @staticmethod
    def _report(stem, mode, status, input_path, output_path, notes, warnings_list, presence=None):
        counts = {}
        for note in notes:
            counts[note.decision] = counts.get(note.decision, 0) + 1
        return StemValidationReport(stem=stem, mode=mode, status=status,
                                    input_midi_path=str(input_path),
                                    output_midi_path=str(output_path) if output_path is not None else None,
                                    notes=notes, counts=counts, warnings=warnings_list, presence=presence)


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
