from pathlib import Path
import os
from torch.utils.data import Dataset
from PIL import Image


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
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
