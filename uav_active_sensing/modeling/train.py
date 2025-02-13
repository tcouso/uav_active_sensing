import typer
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from uav_active_sensing.config import PROCESSED_DATA_DIR
from uav_active_sensing.pytorch_dataloaders import TinyImageNetDataset
from uav_active_sensing.modeling.img_exploration_env import RewardFunction
from uav_active_sensing.modeling.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.modeling.ppo import train_ppo


app = typer.Typer()


@app.command()
def main():
    TINY_IMAGENET_PROCESSED_DIR = PROCESSED_DATA_DIR / "tiny_imagenet/tiny-imagenet-200"

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)
    tiny_imagenet_train_dataset = TinyImageNetDataset(root_dir=TINY_IMAGENET_PROCESSED_DIR, split="train", transform=image_processor)
    tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=1, shuffle=True)

    model = ActViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    reward_function = RewardFunction(model)

    for img_batch in tiny_imagenet_train_loader:
        train_ppo(img_batch, reward_function)


if __name__ == "__main__":
    app()
