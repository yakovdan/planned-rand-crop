import copy
from typing import Any, Dict, Optional, Sequence, Union
from torch.utils.data import Dataset
from .types import CropKey


class PlannedRandCropDataset(Dataset):
    """
    Wrap a base dataset and create a dataset of planned random crops where crops are ordered
    according to one of three possible strategies: randomize all crops, shuffle cases and randomize within case
    or shuffle cases and interleave crops from each case.

    Typical use: base_ds is a MONAI CacheDataset/PersistentDataset that returns a dict
    like {"image": ..., "label": ...} with deterministic (non-random) transforms applied.

    Notes:
      - key is a CropKey (case_idx, crop_idx, seed), not an int.
      - crop_t should be configured with num_samples=1. This is critical
    """

    def __init__(
        self,
        base_ds: Dataset,
        crop_t: Any,
        post_t: Optional[Any] = None,
        *,
        deepcopy_case: bool = True,
    ):
        self.base_ds = base_ds
        self.crop_t = crop_t
        self.post_t = post_t
        self.deepcopy_case = deepcopy_case
        if not hasattr(self.crop_t, "set_random_state"):
            raise ValueError("crop_t must support set_random_state(seed) method")

    def __len__(self) -> int:
        # Real iteration length is controlled by the sampler.
        return len(self.base_ds)

    def __getitem__(self, key: CropKey) -> Dict[str, Any]:
        case = self.base_ds[key.case_idx]

        # Avoid mutating cached dict/metatensors in-place. Important for CacheDataset!
        if self.deepcopy_case:
            case = copy.deepcopy(case)

        self.crop_t.set_random_state(seed=key.seed)
        out = self.crop_t(case)

        # Some MONAI transforms may return a list even if num_samples=1
        if isinstance(out, (list, tuple)):
            out = out[0]

        # Attach metadata
        out["case_idx"] = key.case_idx
        out["crop_idx"] = key.crop_idx
        out["seed"] = key.seed

        crop_center = self.crop_t.cropper.centers[0]
        out["crop_center"] = crop_center

        if self.post_t is not None:
            out = self.post_t(out)

        return out
