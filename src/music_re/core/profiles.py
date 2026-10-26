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

    def to_dict(self):
        return asdict(self)


def resolve_profile(name: str, six_stems: bool = False) -> PipelineSettings:
    if name == 'fast':
        return PipelineSettings(name, False, 0, .1, 'librosa', None, False, False)
    if name == 'balanced':
        return PipelineSettings(name, six_stems, 1, .25, 'allin1', 'harmonix-fold0', True, True)
    if name == 'quality':
        return PipelineSettings(name, six_stems, 2, .5, 'allin1', 'harmonix-all', True, True)
    raise ValueError(f"Unknown execution profile: {name}")
