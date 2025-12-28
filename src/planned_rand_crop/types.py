from dataclasses import dataclass

@dataclass(frozen=True)
class CropKey:
    """Index/key used by EpochCropKeySampler and consumed by PlannedRandCropDataset."""
    case_idx: int
    crop_idx: int
    seed: int