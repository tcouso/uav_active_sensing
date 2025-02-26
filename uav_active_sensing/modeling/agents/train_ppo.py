import typer
import mlflow
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from uav_active_sensing.pytorch_datasets import TinyImageNetDataset, tiny_imagenet_collate_fn
from uav_active_sensing.modeling.img_env.img_exploration_env import RewardFunction, ImageExplorationEnv, ImageExplorationEnvConfig
from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.modeling.agents.rl_agent_feature_extractor import CustomResNetFeatureExtractor
from uav_active_sensing.config import DEVICE, SEED

from dataclasses import dataclass

# TODO: Implement a callback for tracking methics and logging them on SB3


class MLflowCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for key, value in self.logger.name_to_value.items():
            mlflow.log_metric(key, value, step=self.num_timesteps)

        return True


@dataclass
class PPOConfig:
    policy: str = "CnnPolicy"
    n_steps: int = None  # Number of steps before learning. A multiple of env.max_steps is recommended
    batch_size: int = None  # Mini-batch size. A factor of n_steps is recommended
    n_epochs: int = 1
    device: str = DEVICE
    seed: int = SEED
    verbose: int = 1
    policy_kwargs: dict = None  # Additional policy arguments


mlflow.autolog()
app = typer.Typer()


@app.command()
def train_ppo(dataset_path: Path = None, model_path: Path = None, img_processor_path: Path = None):

    training_generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    mlflow.set_experiment("test_ppo_training")

    with mlflow.start_run():

        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)  # TODO: Download this in advance
        tiny_imagenet_train_dataset = TinyImageNetDataset(split="train", transform=image_processor)

        # Use worker_init_fn if loading data in multiprocessing settings: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset,
                                                batch_size=2,
                                                collate_fn=tiny_imagenet_collate_fn,
                                                generator=training_generator,
                                                shuffle=True)

        # Pretrained model and reward function
        mae_model = ActViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")  # TODO: Download this in advance
        reward_function = RewardFunction(mae_model)

        # Create a dummy environment to initialize the model
        dummy_batch = next(iter(tiny_imagenet_train_loader))  # Take one image as a dummy input for env initialization
        env_config = ImageExplorationEnvConfig(img=dummy_batch, reward_function=reward_function)
        env = ImageExplorationEnv(env_config)

        ppo_agent_policy_kwargs = dict(
            features_extractor_class=CustomResNetFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
        )

        ppo_config = PPOConfig(
            n_steps=env.max_steps * 2,
            batch_size=env.max_steps,  # Ensures one gradient update per learn()
            policy_kwargs=ppo_agent_policy_kwargs,
        )

        # Log ppo configuration params
        mlflow.log_params(vars(ppo_config))
        mlflow_callback = MLflowCallback()

        ppo_agent = PPO(
            ppo_config.policy,
            env,
            policy_kwargs=ppo_config.policy_kwargs,
            verbose=ppo_config.verbose,
            device=ppo_config.device,
            seed=ppo_config.seed,
            n_steps=ppo_config.n_steps,
            batch_size=ppo_config.batch_size,
            n_epochs=ppo_config.n_epochs,
        )

        logger.info("Starting batch iterations")
        for i, batch in enumerate(tiny_imagenet_train_loader):
            vec_env = ppo_agent.get_env()
            vec_env.env_method("set_img", batch)
            ppo_agent.learn(total_timesteps=2 * ppo_config.n_steps, progress_bar=False, callback=mlflow_callback)

            mlflow.log_metric(f"batch", i)

            if i == 5:
                break

            logger.info(f"Batch {i} completed")


if __name__ == "__main__":
    app()
