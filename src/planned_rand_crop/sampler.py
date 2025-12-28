import random
from typing import Iterator, Literal
from torch.utils.data import Sampler
from .types import CropKey

class EpochCropKeySampler(Sampler[CropKey]):
    """
    Produce a full-epoch plan of CropKeys.

    shuffle_mode:
      - "all_crops": global shuffle among all crops in the epoch
      - "cases_then_crops": shuffle cases, keep each case's crops grouped
      - "interleave_by_crop": shuffle cases, then interleave crops (one per case, repeated)
    """

    def __init__(self, num_cases: int, crops_per_case: int, *, base_seed: int = 0,
                 shuffle_mode: Literal["all_crops", "cases_then_crops", "interleave_by_crop"] = "all_crops",
                 shuffle_within_case: bool = False):
        super().__init__()
        if num_cases <= 0:
            raise ValueError("num_cases must be > 0")
        if crops_per_case <= 0:
            raise ValueError("crops_per_case must be > 0")

        self.num_cases = num_cases
        self.crops_per_case = crops_per_case
        self.base_seed = base_seed
        self.shuffle_mode = shuffle_mode
        self.shuffle_within_case = shuffle_within_case
        self.epoch = 0
        self.last_epoch = None

    def set_epoch(self, epoch: int) -> None:
        """Call this at the start of each epoch (like DistributedSampler)."""
        self.last_epoch = self.epoch
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[CropKey]:
        if self.last_epoch and self.last_epoch == self.epoch:
            raise RuntimeError("Sampler epoch not advanced; call set_epoch(epoch) with new epoch before iterating again.")

        rng = random.Random(self.base_seed + self.epoch)

        case_order = list(range(self.num_cases))
        rng.shuffle(case_order)

        keys_by_case: list[list[CropKey]] = []
        for ci in case_order:
            seeds = [rng.getrandbits(32) for _ in range(self.crops_per_case)]
            crop_ids = list(range(self.crops_per_case))
            if self.shuffle_within_case:
                rng.shuffle(crop_ids)
            keys_by_case.append([CropKey(ci, cj, seeds[cj]) for cj in crop_ids])

        if self.shuffle_mode == "cases_then_crops":
            plan = [k for case_keys in keys_by_case for k in case_keys]

        elif self.shuffle_mode == "interleave_by_crop":
            plan: list[CropKey] = []
            for j in range(self.crops_per_case):
                for case_keys in keys_by_case:
                    plan.append(case_keys[j])

        elif self.shuffle_mode == "all_crops":
            plan = [k for case_keys in keys_by_case for k in case_keys]
            rng.shuffle(plan)

        else:
            raise ValueError(f"Unknown shuffle_mode={self.shuffle_mode}")

        return iter(plan)

    def __len__(self) -> int:
        return self.num_cases * self.crops_per_case
