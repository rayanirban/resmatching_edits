from pathlib import Path

import numpy as np
import torch
from torch.utils import data
from tifffile import imread

from .data_norm import normalize

SUBSETS = ["ccp", "er", "factin", "mt", "mt_noisy"]


class BioSRDataset(data.Dataset):
    """BioSR dataset for paired low-res / high-res fluorescence microscopy images.

    Args:
        subset: One of ``"ccp"``, ``"er"``, ``"factin"``, ``"mt"``, ``"mt_noisy"``.
        folder: Directory containing image crops.
        returns: Channel indices to include in each sample, e.g. ``[0, 1]``.
    """

    def __init__(
        self,
        subset: str,
        folder: Path,
        returns: list[int] = [0, 1],
    ):
        super().__init__()
        if subset not in SUBSETS:
            raise ValueError(f"subset must be one of {SUBSETS}, got {repr(subset)}")

        self.subset = subset
        self.returns = returns

        self.paths = sorted(Path(folder).glob("*.tif"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = torch.from_numpy(imread(self.paths[index]).astype(np.float32))
        channels = [normalize(img[ch : ch + 1], self.subset, ch) for ch in self.returns]
        return torch.cat(channels, dim=0)
