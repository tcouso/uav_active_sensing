import typer
import random as rd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pathlib import Path
import sys
from typing import Dict, Union, Any, Tuple
import mlflow
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ViTMAEForPreTraining
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from uav_active_sensing.pytorch_datasets import TinyImageNetDataset, tiny_imagenet_collate_fn
from uav_active_sensing.modeling.img_env.img_exploration_env import RewardFunction, ImageExplorationEnv, ImageExplorationEnvConfig
from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.modeling.agents.rl_agent_feature_extractor import CustomResNetFeatureExtractor
from uav_active_sensing.config import DEVICE, SEED
from uav_active_sensing.plots import visualize_act_mae_reconstruction, visualize_mae_reconstruction

app = typer.Typer()

PPO_PARAMS = {
    'steps_until_termination': 50,
    'learning_rate': 3e-5,
    'n_steps': 2048,
    'batch_size': 256,
    'n_epochs': 10,
    'clip_range': 0.2,
    'gamma': 0.99,
    'policy': 'CnnPolicy',
    'gae_lambda': 0.95,
    'ent_coef': 0.05,
    'vf_coef': 0.5,
    'device': DEVICE,
    'seed': SEED,
}

PPO_PARAMS_DEBUG = {
    'steps_until_termination': 50,
    'learning_rate': 3e-5,
    'n_steps': 64,
    'batch_size': 16,
    'n_epochs': 10,
    'clip_range': 0.2,
    'gamma': 0.99,
    'policy': 'CnnPolicy',
    'gae_lambda': 0.95,
    'ent_coef': 0.05,
    'vf_coef': 0.5,
    'device': DEVICE,
    'seed': SEED,
}


class RandomAgent:
    def __init__(self, env: ImageExplorationEnv):
        self.action_space = env.action_space

    def predict(self, obs, deterministic=False):
        _, _ = obs, deterministic  # Dummy inputs

        return self.action_space.sample(), None

# Debugging dataset for single image experiment


class SingleImageDataset(Dataset):
    def __init__(self, original_dataset: Dataset, index: int):
        self.image, self.label = original_dataset[index]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.image, self.label


class ImgReconstructinoCallback(BaseCallback):

    def __init__(self, img_reconstruction_period: int):
        super().__init__()
        self.img_reconstruction_period: int = img_reconstruction_period

    def _on_step(self) -> True:
        if self.num_timesteps % self.img_reconstruction_period == 0:
            # Image reconstruction
            pass
        return True


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


def run_episode_and_visualize_sampling(
    ppo_agent: PPO,
    env: ImageExplorationEnv,
    deterministic: bool,
    act_mae_model: ActViTMAEForPreTraining,
    reconstruction_dir: Path,
    filename: str,
    img_index: int,
    mae_model: ViTMAEForPreTraining = None,
):
    """
    Runs one episode with the given agent and environment, then visualizes the reconstructions.
    """
    obs, _ = env.reset()
    done = False

    while not done:
        actions, _ = ppo_agent.predict(
            obs,
            deterministic=deterministic,
        )
        obs, _, done, _, _ = env.step(actions, eval=True)

    visualize_act_mae_reconstruction(
        env.img,
        env.sampled_img,
        act_mae_model,
        show=False,
        save_path=reconstruction_dir / f"{filename}_{img_index}"
    )

    if mae_model is not None:
        visualize_mae_reconstruction(
            env.img,
            mae_model,
            show=False,
            save_path=reconstruction_dir / f"mae_reconstruction_img_{img_index}"
        )


class ImgReconstructinoCallback(BaseCallback):
    def __init__(self, img_reconstruction_period: int,
                 env: ImageExplorationEnv,
                 act_mae_model: ActViTMAEForPreTraining,
                 mae_model: ViTMAEForPreTraining,
                 reconstruction_dir: Path,
                 deterministic: bool = False):

        super().__init__()
        self.img_reconstruction_period: int = img_reconstruction_period
        self.env = env
        self.act_mae_model = act_mae_model
        self.mae_model = mae_model
        self.reconstruction_dir = reconstruction_dir
        self.deterministic = deterministic

    def _on_step(self) -> True:
        if self.num_timesteps % self.img_reconstruction_period == 0:
            # Image reconstruction
            run_episode_and_visualize_sampling(
                ppo_agent=self.model,
                env=self.env,
                deterministic=self.deterministic,
                act_mae_model=self.act_mae_model,
                reconstruction_dir=self.reconstruction_dir,
                filename="ppo_agent",
                img_index=self.num_timesteps,
            )

        return True


# TODO: Implement an image epoch loop after hiperparam search
def train_ppo(params: dict, experiment_name: str = None, nested: bool = False) -> dict:
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(nested=nested):
        mlflow.transformers.autolog(disable=True)
        # mlflow.sklearn.autolog(disable=True)
        mlflow.autolog()

        run_id = mlflow.active_run().info.run_id
        experiment_id = mlflow.active_run().info.experiment_id

        run_dir = Path(f"mlruns/{experiment_id}/{run_id}")
        artifact_dir = run_dir / "artifacts"
        models_dir = artifact_dir / "models"
        logs_dir = artifact_dir / "logs"
        eval_img_reconstruction_dir = artifact_dir / "eval_img_reconstruction_dir"
        train_img_reconstruction_dir = artifact_dir / "train_img_reconstruction_dir"

        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        eval_img_reconstruction_dir.mkdir(parents=True, exist_ok=True)
        train_img_reconstruction_dir.mkdir(parents=True, exist_ok=True)

        torch_generator = torch.Generator().manual_seed(SEED)
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)
        tiny_imagenet_train_dataset = TinyImageNetDataset(split="train", transform=image_processor)

        # Single image experiment debugging
        rd.seed(SEED)
        random_index = rd.randint(0, len(tiny_imagenet_train_dataset) - 1)
        single_image_dataset = SingleImageDataset(tiny_imagenet_train_dataset, random_index)

        # Use worker_init_fn if loading data in multiprocessing settings: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        dataloader = DataLoader(single_image_dataset,
                                batch_size=1,
                                collate_fn=tiny_imagenet_collate_fn,
                                generator=torch_generator,
                                shuffle=True)

        # Pretrained model and reward function
        mae_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(DEVICE)
        act_mae_model = ActViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(DEVICE)
        reward_function = RewardFunction(act_mae_model)

        # Take one image as a dummy input for env initialization
        dummy_batch = next(iter(dataloader))
        env_config = ImageExplorationEnvConfig(img=dummy_batch,
                                               steps_until_termination=params['steps_until_termination'],
                                               reward_function=reward_function,
                                               )
        env = ImageExplorationEnv(env_config)
        ppo_agent_policy_kwargs = dict(
            features_extractor_class=CustomResNetFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
        )
        img_reconstruction_callback = ImgReconstructinoCallback(
            img_reconstruction_period=10_000,
            env=env,
            act_mae_model=act_mae_model,
            mae_model=mae_model,
            reconstruction_dir=train_img_reconstruction_dir,
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
            verbose=1
        )
        rd_agent = RandomAgent(env)
        mlflow.log_params(params)

        loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )
        ppo_agent.set_logger(loggers)
        vec_env = ppo_agent.get_env()

        for i, batch in enumerate(dataloader):
            vec_env.env_method("set_img", batch)
            ppo_agent.learn(total_timesteps=100 * params['n_steps'], progress_bar=False, log_interval=1, callback=img_reconstruction_callback)

        ppo_agent.save(models_dir / "ppo_model.zip")

        # Register the model in MLflow Model Registry
        model_uri = f"runs:/{run_id}/models"
        mlflow.register_model(model_uri, name=f"SB3_PPO_Model_{experiment_id}_{run_id}")

        # Evaluation loop (in same img for single img experiment)
        # tiny_imagenet_val_dataset = TinyImageNetDataset(split="val", transform=image_processor)
        # tiny_imagenet_val_loader = DataLoader(tiny_imagenet_val_dataset,
        #                                       batch_size=env_config.img_batch_size,
        #                                       collate_fn=tiny_imagenet_collate_fn,
        #                                       generator=torch_generator,
        #                                       shuffle=True)

        eval_loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )
        ppo_agent.set_logger(eval_loggers)

        total_mean_reward = 0
        val_batch_count = 0
        reward_list = []
        for i, batch in enumerate(dataloader):
            vec_env.env_method("set_img", batch)
            mean_reward, _ = evaluate_policy(
                ppo_agent,
                vec_env,
                n_eval_episodes=100,
                deterministic=True,
                return_episode_rewards=False
            )

            total_mean_reward += mean_reward
            reward_list.append(mean_reward)
            val_batch_count += 1

            # if i % (len(tiny_imagenet_val_loader.dataset) // 10) == 0:
            run_episode_and_visualize_sampling(
                ppo_agent,
                env,
                deterministic=True,
                act_mae_model=act_mae_model,
                mae_model=mae_model,
                filename="ppo_agent",
                reconstruction_dir=eval_img_reconstruction_dir,
                img_index=i,
            )
            run_episode_and_visualize_sampling(
                rd_agent,
                env,
                deterministic=True,
                act_mae_model=act_mae_model,
                mae_model=mae_model,
                filename="rd_agent",
                reconstruction_dir=eval_img_reconstruction_dir,
                img_index=i,
            )

        val_mean_reward = total_mean_reward / val_batch_count
        val_std_reward = np.std(reward_list, ddof=1) if val_batch_count > 1 else 0

        mlflow.log_metric("eval/mean_reward", val_mean_reward)
        mlflow.log_metric("eval/std_reward", val_std_reward)

        return {'loss': -val_mean_reward,
                'loss_variance': -val_std_reward,
                "status": STATUS_OK}


def objective(params: dict) -> dict:
    result = train_ppo(params, nested=True)

    return result


@app.command()
def ppo_fixed_params(experiment_name: str):
    train_ppo(PPO_PARAMS, experiment_name)


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
        'steps_until_termination': hp.choice('steps_until_termination', [10, 20, 30]),  # Fewer steps
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(5e-4)),  # Higher floor, smaller range
        'n_steps': hp.choice('n_steps', [30, 40, 50]),  # Much smaller update steps
        'batch_size': hp.choice('batch_size', [8, 16]),  # Smaller batch sizes
        'n_epochs': hp.choice('n_epochs', [1, 2]),  # Fewer epochs
        'clip_range': hp.uniform('clip_range', 0.2, 0.3),  # Keep range small
        'gamma': 0.9,  # Lower discount factor for quicker updates
        'policy': 'CnnPolicy',
        'gae_lambda': 0.8,  # Lower GAE lambda for faster updates
        'ent_coef': 0.0,
        'vf_coef': 0.2,  # Reduce value function coefficient to prioritize speed
        'device': DEVICE,
        'seed': SEED,  # Fixed seed for reproducibility
    }
    with mlflow.start_run():
        # Conduct the hyperparameter search using Hyperopt
        trials = Trials()
        best = fmin(
            fn=objective,
            space=param_space,
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
    # Remember to start server in cli from root dir: ```mlflow server --host 0.0.0.0 --port 5000```
    app()
