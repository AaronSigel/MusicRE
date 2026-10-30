from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PipelineSettings:
    profile: str
    six_stems: bool
    shifts: int
    overlap: float
    structure_backend: str
    structure_model: str | None
    save_raw: bool
    evaluate: bool
    grid_subdivisions: int = 4
    quantize_ends: bool = True
    seed: int = 0
    stems_count: int = 4
    model_name: str = "htdemucs"

    def __post_init__(self):
        object.__setattr__(self, 'stems_count', 6 if self.six_stems else 4)
        object.__setattr__(self, 'model_name', 'htdemucs_6s' if self.six_stems else 'htdemucs')

    @property
    def supported_stems(self) -> list[str]:
        if self.six_stems:
            return ['vocals', 'drums', 'bass', 'other', 'guitar', 'piano']
        return ['vocals', 'drums', 'bass', 'other']

    def to_dict(self):
        return asdict(self)


def resolve_profile(name: str, six_stems: bool = False) -> PipelineSettings:
    key = name.lower()
    if key in ('dense', 'six_stems', 'six-stems', 'quality_6s', 'quality-6s'):
        return PipelineSettings(name, True, 2, .5, 'allin1', 'harmonix-all', True, True)
    if key == 'fast':
        return PipelineSettings(name, six_stems, 0, .1, 'librosa', None, False, False)
    if key == 'balanced':
        return PipelineSettings(name, six_stems, 1, .25, 'allin1', 'harmonix-fold0', True, True)
    if key == 'quality':
        return PipelineSettings(name, six_stems, 2, .5, 'allin1', 'harmonix-all', True, True)
    raise ValueError(f"Unknown execution profile: {name}")
