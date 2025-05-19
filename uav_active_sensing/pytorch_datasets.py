from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

from uav_active_sensing.config import DEVICE, DATA_DIR

# Tiny imagenet


def tiny_imagenet_collate_fn(batch):
    processed_batch = [img[0]["pixel_values"].to(DEVICE) for img in batch]

    return torch.cat(processed_batch, dim=0)


def tiny_imagenet_single_img_collate_fn(batch):
    processed_batch = [img[0]["pixel_values"].to(DEVICE) for img in batch]
    img = processed_batch[0].squeeze(0)
    assert len(img.shape) == 3, "Expect batch size of 1"

    return img


class TinyImageNetDataset(Dataset):
    def __init__(
        self, root_dir=DATA_DIR / "tiny-imagenet-200", split=None, transform=None
    ):
        """
        Args:
            root_dir (Path or str): Root directory of the dataset (tiny-imagenet-200).
            split (str): 'train' or 'val'.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.data = []
        self.labels = []

        if split == "train":
            for class_dir in os.listdir(self.root_dir):
                class_path = self.root_dir / class_dir / "images"
                for img_file in os.listdir(class_path):
                    self.data.append(class_path / img_file)
                    self.labels.append(class_dir)
        elif split == "val":
            val_annotations = self.root_dir / "val_annotations.txt"
            with open(val_annotations, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img_file, class_label = parts[0], parts[1]
                    self.data.append(self.root_dir / "images" / img_file)
                    self.labels.append(class_label)

        # Map labels to indices
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
