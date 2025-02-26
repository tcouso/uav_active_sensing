import typer
from pathlib import Path
import sys
from typing import Dict, Union, Any, Tuple
import mlflow
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger

from uav_active_sensing.pytorch_datasets import TinyImageNetDataset, tiny_imagenet_collate_fn
from uav_active_sensing.modeling.img_env.img_exploration_env import RewardFunction, ImageExplorationEnv, ImageExplorationEnvConfig
from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.modeling.agents.rl_agent_feature_extractor import CustomResNetFeatureExtractor
from uav_active_sensing.config import DEVICE, SEED


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


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


app = typer.Typer()


@app.command()
def train_ppo(experiment_name: str):
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run():
        mlflow.autolog()

        run_id = mlflow.active_run().info.run_id
        experiment_id = mlflow.active_run().info.experiment_id

        run_dir = Path(f"mlruns/{experiment_id}/{run_id}")
        artifact_dir = run_dir / "artifacts"
        models_dir = Path(artifact_dir / "models")
        logs_dir = Path(artifact_dir / "logs")

        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        training_generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)
        tiny_imagenet_train_dataset = TinyImageNetDataset(split="train", transform=image_processor)

        # Use worker_init_fn if loading data in multiprocessing settings: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset,
                                                batch_size=2,
                                                collate_fn=tiny_imagenet_collate_fn,
                                                generator=training_generator,
                                                shuffle=True)

        # Pretrained model and reward function
        mae_model = ActViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        reward_function = RewardFunction(mae_model)

        # Take one image as a dummy input for env initialization
        dummy_batch = next(iter(tiny_imagenet_train_loader))
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
        mlflow.log_params(vars(ppo_config))

        loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )
        ppo_agent.set_logger(loggers)

        for i, batch in enumerate(tiny_imagenet_train_loader):
            vec_env = ppo_agent.get_env()
            vec_env.env_method("set_img", batch)
            ppo_agent.learn(total_timesteps=2 * ppo_config.n_steps, progress_bar=False, log_interval=1)

            if i == 1:  # For debugging
                break

        ppo_agent.save(models_dir / "ppo_model.zip")

        # Register the model in MLflow Model Registry
        model_uri = f"runs:/{run_id}/models"
        mlflow.register_model(model_uri, name="SB3_PPO_Model")


if __name__ == "__main__":
    app()
