from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


@dataclass
class FlowerSample:
    image_path: Path
    label: int
    image_id: int


class OxfordFlowers102Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = load_split_samples(self.root, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label,
            "image_id": sample.image_id,
            "image_path": str(sample.image_path),
        }


def load_split_samples(root: str | Path, split: str) -> list[FlowerSample]:
    root = Path(root)
    setid = loadmat(root / "setid.mat")
    labels = loadmat(root / "imagelabels.mat")["labels"].reshape(-1)
    split_key_map = {"train": "trnid", "val": "valid", "test": "tstid"}
    if split not in split_key_map:
        raise ValueError(f"Unsupported split: {split}")
    image_ids = setid[split_key_map[split]].reshape(-1)

    samples: list[FlowerSample] = []
    for image_id in image_ids:
        image_id = int(image_id)
        label = int(labels[image_id - 1]) - 1
        image_path = root / "102flowers" / "jpg" / f"image_{image_id:05d}.jpg"
        samples.append(FlowerSample(image_path=image_path, label=label, image_id=image_id))
    return samples


def get_split_sizes(root: str | Path) -> dict[str, int]:
    return {
        split: len(load_split_samples(root, split))
        for split in ("train", "val", "test")
    }
