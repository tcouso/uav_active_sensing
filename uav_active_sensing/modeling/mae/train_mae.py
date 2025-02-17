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
from uav_active_sensing.modeling.mae.act_vit_mae_config import ActViTMAETrainingConfig
# from uav_active_sensing.modeling.ppo import make_env, PPOConfig, PPOAgent
from uav_active_sensing.utils import load_model_state_dict

# Datasets
image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)
tiny_imagenet_train_dataset = TinyImageNetDataset(split="train", transform=image_processor)
tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=1, collate_fn=tiny_imagenet_collate_fn)

app = typer.Typer()

# TODO: Update to SB3 Agents version

@app.command()
def train_mae():
    trained_model_state_dict = load_model_state_dict(MODELS_DIR / "runs/ImageExploration-v0__ppo__42__1739429656/ppo.cleanrl_model", DEVICE)  # TODO: model path as param
    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)  # TODO: load pretrained models locally
    tiny_imagenet_train_dataset = TinyImageNetDataset(split="train", transform=image_processor)
    tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=IMG_BATCH_SIZE, collate_fn=tiny_imagenet_collate_fn)

    mae_model = ActViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")  # TODO: load pretrained models locally
    reward_function = RewardFunction(mae_model)  # TODO: We could avoid using this in order to not run inference with each step in env

    # Define optimizer (Adam)
    mae_training_config = ActViTMAETrainingConfig()
    optimizer = optim.Adam(
        mae_model.parameters(),
        lr=mae_training_config.base_learning_rate,
        weight_decay=mae_training_config.weight_decay
    )

    # Define learning rate scheduler (cosine annealing with warmup)
    total_steps = len(tiny_imagenet_train_loader) * mae_training_config.num_train_epochs
    warmup_steps = int(mae_training_config.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Epoch for mask generation
    for img_batch in tiny_imagenet_train_loader:

        # # Mask generation with trained agent
        # # TODO: Set an inference mode (with no reward computation)
        # vect_env = gym.vector.SyncVectorEnv(
        #     [make_env(img.unsqueeze(0), reward_function, ppo_config.gamma) for img in img_batch]
        # )
        # trained_agent = None
        # trained_agent.load_state_dict(trained_model_state_dict)
        # trained_agent.eval()

        # global_step = 0
        # next_obs, _ = vect_env.reset(seed=ppo_config.seed)
        # next_obs = torch.Tensor(next_obs).to(DEVICE)
        # next_done = torch.zeros(ppo_config.num_envs).to(DEVICE)

        # # Image sampling
        # for step in range(0, ppo_config.num_steps):
        #     global_step += ppo_config.num_envs

        #     with torch.no_grad():
        #         action, _, _, _ = trained_agent.get_action_and_value(next_obs)
        #         action = torch.round(action).int()

        #     next_obs, _, terminations, truncations, _ = vect_env.step(action.cpu().numpy())
        #     next_done = np.logical_or(terminations, truncations)
        #     next_obs = torch.Tensor(next_obs).to(DEVICE)
        #     next_done = torch.Tensor(next_done).to(DEVICE)
        pass
    
        env = None
        # MAE training with generated masks
        for img, sampled_img in zip(env.img, env.sampled_img):

            optimizer.zero_grad()
            outputs = mae_model(img, sampled_img)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        break
