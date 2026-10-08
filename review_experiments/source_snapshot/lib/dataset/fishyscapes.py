import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Fishyscapes(Dataset):
    """
    Fishyscapes dataset loader.

    Expected directory structure:
        root/
        ├── Static
        │   ├── original
        │   └── labels
        └── LostAndFound
            ├── original
            └── labels

    Labels are expected to already use:
        0   -> inlier
        1   -> OoD
        255 -> ignore
    """

    def __init__(
        self,
        split: str = "Static",
        root: str = "/root/shared-nvme/datasets/fishyscapes",
        transform=None,
    ):
        assert split in ["Static", "LostAndFound"], f"Unsupported split: {split}"

        self.root = root
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, split, "original")
        self.label_dir = os.path.join(root, split, "labels")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.images, self.targets = self._build_pairs()

        if len(self.images) == 0:
            raise RuntimeError(
                f"No valid Fishyscapes pairs found under split={split}, root={root}"
            )

    def _build_pairs(self) -> Tuple[List[str], List[str]]:
        label_files = sorted(
            [
                f
                for f in os.listdir(self.label_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        images: List[str] = []
        targets: List[str] = []

        for lb in label_files:
            stem = os.path.splitext(lb)[0]

            candidates = [
                os.path.join(self.image_dir, stem + ".png"),
                os.path.join(self.image_dir, stem + ".jpg"),
                os.path.join(self.image_dir, stem + ".jpeg"),
                os.path.join(self.image_dir, lb),
            ]

            img_path = None
            for cand in candidates:
                if os.path.exists(cand):
                    img_path = cand
                    break

            if img_path is None:
                continue

            label_path = os.path.join(self.label_dir, lb)
            images.append(img_path)
            targets.append(label_path)

        return images, targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image_path = self.images[i]
        target_path = self.targets[i]

        image = Image.open(image_path).convert("RGB")
        target = Image.open(target_path).convert("L")

        target_np = np.array(target, dtype=np.uint8)

        # Your dataset is already confirmed to use [0, 1, 255].
        # We keep it unchanged for evaluation.
        target = Image.fromarray(target_np, mode="L")

        if self.transform is not None:
            image, target = self.transform(image, target)

        file_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, target, file_name