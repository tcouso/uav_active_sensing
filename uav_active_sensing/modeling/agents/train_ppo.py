import typer
import hyperopt as hp
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
from stable_baselines3.common.evaluation import evaluate_policy

from uav_active_sensing.pytorch_datasets import TinyImageNetDataset, tiny_imagenet_collate_fn
from uav_active_sensing.modeling.img_env.img_exploration_env import RewardFunction, ImageExplorationEnv, ImageExplorationEnvConfig
from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.modeling.agents.rl_agent_feature_extractor import CustomResNetFeatureExtractor
from uav_active_sensing.config import DEVICE, SEED


# TODO: Setup PPO config here, and log params in mlflow. The idea could be iterate over a loop of params
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# TODO: Set scores that can be used to sort experiments (reward per episode and actual image MSE)
# Compare img MSE with MAE MSE


# param_space = {
#     'steps_until_termination': hp.choice('n_steps', [30, 40, 50]),
#     'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)),
#     'n_steps': hp.choice('n_steps', [128, 256, 512]),
#     'batch_size': hp.choice('batch_size', [32, 64, 128]),
#     'n_epochs': hp.choice('n_epochs', [3, 5, 10]),
#     'clip_range': hp.uniform('clip_range', 0.1, 0.3),
#     'gamma': 0.99,
#     'gae_lambda': 0.95,
#     'ent_coef': 0.0,
#     'vf_coef': 0.5,
# }

# def objective(params):
#     train_ppo()
#     result = train_ppo()

#     return result
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


app = typer.Typer()

# TODO: Implement an image epoch loop


@app.command()
def train_ppo(experiment_name: str):

    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run():
        mlflow.transformers.autolog(disable=True)
        mlflow.autolog()

        run_id = mlflow.active_run().info.run_id
        experiment_id = mlflow.active_run().info.experiment_id

        run_dir = Path(f"mlruns/{experiment_id}/{run_id}")
        artifact_dir = run_dir / "artifacts"
        models_dir = Path(artifact_dir / "models")
        logs_dir = Path(artifact_dir / "logs")

        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        torch_generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)
        tiny_imagenet_train_dataset = TinyImageNetDataset(split="train", transform=image_processor)

        # Use worker_init_fn if loading data in multiprocessing settings: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset,
                                                batch_size=1,
                                                collate_fn=tiny_imagenet_collate_fn,
                                                generator=torch_generator,
                                                shuffle=True)

        # Pretrained model and reward function
        mae_model = ActViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        reward_function = RewardFunction(mae_model)

        # Take one image as a dummy input for env initialization
        dummy_batch = next(iter(tiny_imagenet_train_loader))
        env_config = ImageExplorationEnvConfig(img=dummy_batch,
                                               reward_function=reward_function)
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
        vec_env = ppo_agent.get_env()
        for i, batch in enumerate(tiny_imagenet_train_loader):
            vec_env.env_method("set_img", batch)
            ppo_agent.learn(total_timesteps=2 * ppo_config.n_steps, progress_bar=False, log_interval=1)

            if i == 1:  # For debugging
                break

        ppo_agent.save(models_dir / "ppo_model.zip")

        # Register the model in MLflow Model Registry
        model_uri = f"runs:/{run_id}/models"
        mlflow.register_model(model_uri, name=f"SB3_PPO_Model_{experiment_id}_{run_id}")

        # Evaluation loop
        # trained_ppo_agent = PPO.load(models_dir / "ppo_model.zip")
        tiny_imagenet_val_dataset = TinyImageNetDataset(split="val", transform=image_processor)
        tiny_imagenet_val_loader = DataLoader(tiny_imagenet_val_dataset,
                                              batch_size=env_config.img_batch_size,
                                              collate_fn=tiny_imagenet_collate_fn,
                                              generator=torch_generator,
                                              shuffle=True)

        eval_loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )
        ppo_agent.set_logger(eval_loggers)
        for i, batch in enumerate(tiny_imagenet_val_loader):
            vec_env.env_method("set_img", batch)
            mean_reward, std_reward = evaluate_policy(
                ppo_agent,
                vec_env,
                n_eval_episodes=10,
                deterministic=True,
                return_episode_rewards=False
            )
            eval_loggers.record("eval/mean_reward", mean_reward)
            eval_loggers.record("eval/std_reward", std_reward)
            eval_loggers.dump(i)  # Ensure logging at each batch

            if i == 1:  # For debugging
                break


if __name__ == "__main__":
    app()
