import typer
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pathlib import Path
import sys
from typing import Dict, Union, Any, Tuple
import mlflow
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

app = typer.Typer()


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


# TODO: Implement an image epoch loop
def train_ppo(params: dict) -> dict:

    with mlflow.start_run(nested=True):
        mlflow.transformers.autolog(disable=True)
        # mlflow.sklearn.autolog(disable=True)
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

        ppo_agent = PPO(
            params['policy'],
            env,
            policy_kwargs=ppo_agent_policy_kwargs,
            device=params['device'],
            seed=params['seed'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
        )
        mlflow.log_params(params)

        loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )
        ppo_agent.set_logger(loggers)
        vec_env = ppo_agent.get_env()
        for i, batch in enumerate(tiny_imagenet_train_loader):
            vec_env.env_method("set_img", batch)
            ppo_agent.learn(total_timesteps=2 * params['n_steps'], progress_bar=False, log_interval=1)

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

        total_mean_reward = 0
        val_batch_count = 0
        reward_list = []
        for i, batch in enumerate(tiny_imagenet_val_loader):
            vec_env.env_method("set_img", batch)
            mean_reward, _ = evaluate_policy(
                ppo_agent,
                vec_env,
                n_eval_episodes=10,
                deterministic=True,
                return_episode_rewards=False
            )

            total_mean_reward += mean_reward
            reward_list.append(mean_reward)  # Store mean rewards
            val_batch_count += 1

            # TODO: Save some reconstructed images as artifacts, with MSE, and comparison with MAE

            if i == 1:  # For debugging
                break

        val_mean_reward = total_mean_reward / val_batch_count
        val_std_reward = np.std(reward_list, ddof=1) if val_batch_count > 1 else 0

        mlflow.log_metric("eval/mean_reward", val_mean_reward)
        mlflow.log_metric("eval/std_reward", val_std_reward)

        return {'loss': -val_mean_reward,
                'loss_variance': -val_std_reward,
                "status": STATUS_OK}


def mock_objective_function(params):
    import random
    total_mean_reward = random.uniform(0, 10)  # Simulating mean reward
    total_std_reward = random.uniform(0, 0.5)  # Simulating standard deviation
    val_batch_count = random.randint(1, 10)  # Simulating batch count

    return {
        'loss': -(total_mean_reward / val_batch_count),
        'loss_variance': -(total_std_reward / val_batch_count),
        "status": STATUS_OK
    }


def objective(params: dict) -> dict:
    result = train_ppo(params)
    # result = mock_objective_function(params)

    return result


@app.command()
def ppo_param_search(experiment_name: str) -> None:
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://localhost:5000")
    param_space = {
        'steps_until_termination': hp.choice('steps_until_termination', [30, 40, 50]),
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)),
        'n_steps': hp.choice('n_steps', [128, 256, 512]),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'n_epochs': hp.choice('n_epochs', [3, 5, 10]),
        'clip_range': hp.uniform('clip_range', 0.1, 0.3),
        'gamma': 0.99,
        'policy': 'CnnPolicy',
        'gae_lambda': 0.95,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'device': DEVICE,
        'seed': SEED,
    }
    param_space_debug = {
        'steps_until_termination': hp.choice('steps_until_termination', [5, 10]),  # Fewer steps
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(5e-4)),  # Higher floor, smaller range
        'n_steps': hp.choice('n_steps', [8, 16]),  # Much smaller update steps
        'batch_size': hp.choice('batch_size', [8, 16]),  # Smaller batch sizes
        'n_epochs': hp.choice('n_epochs', [1, 2]),  # Fewer epochs
        'clip_range': hp.uniform('clip_range', 0.2, 0.3),  # Keep range small
        'gamma': 0.9,  # Lower discount factor for quicker updates
        'policy': 'CnnPolicy',
        'gae_lambda': 0.8,  # Lower GAE lambda for faster updates
        'ent_coef': 0.0,
        'vf_coef': 0.2,  # Reduce value function coefficient to prioritize speed
        'device': 'cpu',  # Force CPU for debugging
        'seed': 42,  # Fixed seed for reproducibility
    }
    with mlflow.start_run():
        # Conduct the hyperparameter search using Hyperopt
        trials = Trials()
        best = fmin(
            fn=objective,
            space=param_space_debug,
            algo=tpe.suggest,
            max_evals=2,
            trials=trials,
        )

        # Fetch the details of the best run
        best_run = sorted(trials.results, key=lambda x: -x["loss"])[0]

        # Log the best parameters, loss, and model
        mlflow.log_params(best)
        mlflow.log_metric("eval/mean_reward", -best_run["loss"])
        mlflow.log_metric("eval/std_reward", -best_run["loss_variance"])

        # Print out the best parameters and corresponding loss
        print(f"Best parameters: {best}")
        print(f"Best val mean reward: {-best_run['loss']}")


if __name__ == "__main__":
    app()
