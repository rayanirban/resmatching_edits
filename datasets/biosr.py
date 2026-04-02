from pathlib import Path

import numpy as np
import torch
from torch.utils import data
from tifffile import imread

from datasets.data_norm import normalize

EXTENSIONS = ["tif", "tiff", "png", "jpg", "jpeg", "bmp"]

SUBSETS = ["ccp", "er", "factin", "mt", "mt_noisy"]


class BioSRDataset(data.Dataset):
    """BioSR dataset for paired low-res / high-res fluorescence microscopy images.

    Args:
        subset: One of ``"ccp"``, ``"er"``, ``"factin"``, ``"mt"``, ``"mt_noisy"``.
        folder_noisy: Directory containing low-res (noisy) input crops.
        folder_clean: Directory containing high-res (clean) target crops.
        returns: Channel indices to include in each sample, e.g. ``[0, 1]``.
        returns_type: Per-channel source — ``"c"`` reads from the clean folder,
            any other value reads from the noisy folder.
        mode: ``"train"`` or ``"val"`` (currently informational only).
    """

    def __init__(
        self,
        subset: str,
        folder_noisy: Path = Path("data/train_crop/"),
        folder_clean: Path = Path("data/train_crop/"),
        returns: list[int] = [0, 1],
        returns_type: list[str] = ["c", "c"],
        mode: str = "train",
    ):
        super().__init__()
        if subset not in SUBSETS:
            raise ValueError(f"subset must be one of {SUBSETS}, got {repr(subset)}")

        self.subset = subset
        self.mode = mode
        self.returns = returns
        self.returns_type = returns_type

        self.paths_noisy = sorted(
            p for ext in EXTENSIONS for p in Path(folder_noisy).glob(f"**/*.{ext}")
        )
        self.paths_clean = sorted(
            p for ext in EXTENSIONS for p in Path(folder_clean).glob(f"**/*.{ext}")
        )

    def __len__(self):
        return len(self.paths_noisy)

    def __getitem__(self, index):
        img_clean = torch.from_numpy(imread(self.paths_clean[index]).astype(np.float32))
        img_noisy = torch.from_numpy(imread(self.paths_noisy[index]).astype(np.float32))

        channels = []
        for ch, src in zip(self.returns, self.returns_type):
            img = img_clean if src == "c" else img_noisy
            channels.append(normalize(img[ch : ch + 1], self.subset, ch))
        return torch.cat(channels, dim=0)
