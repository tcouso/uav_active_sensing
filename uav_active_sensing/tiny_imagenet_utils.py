import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from loguru import logger

import torch
from torch.utils.data import Dataset

from uav_active_sensing.config import DEVICE, EXTERNAL_DATA_DIR, TINY_IMAGENET_URL


def download_and_extract_tiny_imagenet(target_dir: Path):
    """Downloads and extracts the Tiny ImageNet dataset."""
    zip_path = target_dir / "tiny-imagenet-200.zip"

    if not zip_path.exists():
        logger.info("Downloading Tiny ImageNet dataset...")
        response = requests.get(TINY_IMAGENET_URL, stream=True)
        with open(zip_path, "wb") as f:
            total_size = int(response.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
        logger.info("Download completed.")

    extracted_dir = target_dir / "tiny-imagenet-200"
    if not extracted_dir.exists():
        logger.info("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        logger.info(f"Dataset extracted to {extracted_dir}")
    else:
        logger.info(f"Tiny ImageNet dataset already extracted at {extracted_dir}")


def tiny_imagenet_collate_fn(batch):
    processed_batch = [img[0]["pixel_values"].to(DEVICE) for img in batch]

    return torch.cat(processed_batch, dim=0)


def tiny_imagenet_single_img_collate_fn(batch):
    processed_batch = [img[0]["pixel_values"].to(DEVICE) for img in batch]
    img = processed_batch[0].squeeze(0)
    assert len(img.shape) == 3, "Expect batch size of 1"

    return img


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir=EXTERNAL_DATA_DIR / "tiny-imagenet-200", split=None, transform=None):
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
