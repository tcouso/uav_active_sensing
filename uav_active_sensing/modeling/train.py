import typer
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AutoImageProcessor

from uav_active_sensing.pytorch_datasets import TinyImageNetDataset, tiny_imagenet_collate_fn


# Datasets
image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)
tiny_imagenet_train_dataset = TinyImageNetDataset(split="train", transform=image_processor)
tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=1, collate_fn=tiny_imagenet_collate_fn)

app = typer.Typer()



@app.command()
def train_reward_mae():

    # TODO: Cycles between rl agent training and MAE training

    pass


if __name__ == "__main__":
    app()
