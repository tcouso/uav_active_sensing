import typer
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoImageProcessor
import gymnasium as gym
from stable_baselines3 import PPO

from uav_active_sensing.config import IMG_BATCH_SIZE, DEVICE, MODELS_DIR
from uav_active_sensing.pytorch_datasets import TinyImageNetDataset, tiny_imagenet_collate_fn
from uav_active_sensing.modeling.img_env.img_exploration_env import RewardFunction, img_pairs_generator, ImageExplorationEnv, ImageExplorationEnvConfig
from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.modeling.mae.configuration_act_vit_mae import ActViTMAETrainingConfig
# from uav_active_sensing.modeling.ppo import make_env, PPOConfig, PPOAgent
from uav_active_sensing.utils import load_model_state_dict

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
