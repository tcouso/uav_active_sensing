import typer

from uav_active_sensing.config import PROCESSED_DATA_DIR

from uav_active_sensing.pytorch_dataloaders import TinyImageNetDataset
from uav_active_sensing.img_exploration_env import ImageExplorationEnv, RewardFunction
from uav_active_sensing.modeling.act_vit_mae import ActViTMAEForPreTraining


from torch.utils.data import DataLoader
from transformers import AutoImageProcessor


app = typer.Typer()


@app.command()
def main():
    TINY_IMAGENET_PROCESSED_DIR = PROCESSED_DATA_DIR / "tiny_imagenet/tiny-imagenet-200"

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)
    custom_model = ActViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    reward_function = RewardFunction(custom_model)

    tiny_imagenet_train_dataset = TinyImageNetDataset(root_dir=TINY_IMAGENET_PROCESSED_DIR, split="train")
    # tiny_imagenet_val_dataset = TinyImageNetDataset(
    #     root_dir=TINY_IMAGENET_PROCESSED_DIR, split="val", transform=image_processor
    # )

    # tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=1, shuffle=True)
    # tiny_imagenet_val_loader = DataLoader(tiny_imagenet_val_dataset, batch_size=1, shuffle=False)

    image, _ = tiny_imagenet_train_dataset[15080]

    inputs = image_processor(images=image, return_tensors="pt")
    env = ImageExplorationEnv(inputs.pixel_values, reward_function, config=custom_model.config)

    rewards = []
    total_reward = 0
    terminated = False

    observation, info = env.reset()

    # Random walk loop for a single image
    while not terminated:
        action = env.action_space.sample()

        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward.detach().numpy()

        if env.step_count % env.interval_reward_assignment == 0:
            print(f"Reward={total_reward}")
            rewards.append(total_reward)
            total_reward = 0

        observation = next_observation

    env.close()


if __name__ == "__main__":
    app()
