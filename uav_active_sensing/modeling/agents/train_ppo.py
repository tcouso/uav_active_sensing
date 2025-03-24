import typer
import random as rd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pathlib import Path
import sys
from typing import Dict, Union, Any, Tuple, Callable
import mlflow
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ViTMAEForPreTraining

from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from uav_active_sensing.pytorch_datasets import TinyImageNetDataset, tiny_imagenet_collate_fn
from uav_active_sensing.modeling.img_env.img_exploration_env import RewardFunction, ImageExplorationEnv, ImageExplorationEnvConfig
from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.modeling.agents.rl_agent_feature_extractor import CustomResNetFeatureExtractor
from uav_active_sensing.config import DEVICE
from uav_active_sensing.plots import visualize_act_mae_reconstruction, visualize_mae_reconstruction

app = typer.Typer()

PPO_PARAMS = {
    'steps_until_termination': 16,
    'interval_reward_assignment': 16,
    'num_samples': 1,
    'masking_ratio': 0.5,
    'reward_increase': False,
    'mask_sample': False,
    'sensor_size': 3 * 16,
    'patch_size': 16,
    'learning_rate': 1e-4,
    'n_steps': 48,
    'total_timesteps': 100_000,
    'batch_size': 48 * 4,
    'num_envs': 1,
    'n_epochs': 3,
    'img_change_period': 16,  # Change image every episode to provide greater generalization
    'clip_range': 0.2,
    'gamma': 0.99,
    'policy': 'MultiInputPolicy',
    'gae_lambda': 0.95,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'device': DEVICE,
    'seed': 64553,
    'img_reconstruction_period': 10_000
}


def run_episode_and_visualize_sampling(
    agent: PPO,
    env: ImageExplorationEnv,
    deterministic: bool,
    mask_sample: bool,
    act_mae_model: ActViTMAEForPreTraining,
    reconstruction_dir: Path,
    filename: str,
    img_index: int,
):
    """
    Runs one episode with the given agent and environment, then visualizes the reconstructions.
    """
    obs, _ = env.reset()
    done = False

    while not done:
        actions, _ = agent.predict(
            obs,
            deterministic=deterministic,
        )
        obs, _, done, _, _ = env.step(actions)

    masked_sampled_img = env._reward_function.sampled_img_random_masking(env.sampled_img)

    if mask_sample:
        visualize_act_mae_reconstruction(
            env.img.unsqueeze(0),
            env.sampled_img.unsqueeze(0),
            masked_sampled_img.unsqueeze(0),
            act_mae_model,
            show=False,
            save_path=reconstruction_dir / f"{filename}_img={img_index}"
        )
    else:  # Sampled image is equal to masked image
        visualize_act_mae_reconstruction(
            env.img.unsqueeze(0),
            env.sampled_img.unsqueeze(0),
            env.sampled_img.unsqueeze(0),
            act_mae_model,
            show=False,
            save_path=reconstruction_dir / f"{filename}_img={img_index}"
        )


class ImageEnvFactory:
    def __init__(self, images: torch.Tensor,
                 log_dir: str,
                 env_config: ImageExplorationEnvConfig):
        self.images = images
        self.log_dir = log_dir
        self.env_config = env_config

    def __call__(self, env_idx: int) -> Callable:
        def _init():
            return Monitor(ImageExplorationEnv(self.images[env_idx], self.env_config), self.log_dir)
        return _init


class SmallImageDataset(Dataset):
    def __init__(self,
                 original_dataset: Dataset,
                 num_images: int):
        """
        Args:
            original_dataset (Dataset): The full dataset.
            num_images (int): The number of images to use for testing.
        """
        self.original_dataset = original_dataset
        self.num_images = num_images  # Parametrized number of images

    def __len__(self):
        return self.num_images  # Return the number of images for testing

    def __getitem__(self, idx):
        # Get the image and label from the original dataset
        img, label = self.original_dataset[idx]
        return img, label


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


# Callbacks
class ImgReconstructinoCallback(BaseCallback):
    def __init__(self,
                 img_reconstruction_period: int,
                 mae_model: ViTMAEForPreTraining,
                 act_mae_model: ActViTMAEForPreTraining,
                 reconstruction_dir: Path,
                 deterministic: bool = False,
                 mask_sample: bool = False,
                 ):

        super().__init__()
        self.img_reconstruction_period: int = img_reconstruction_period
        self.mae_model = mae_model
        self.act_mae_model = act_mae_model
        self.reconstruction_dir = reconstruction_dir
        self.deterministic = deterministic
        self.mask_sample = mask_sample

    def _on_step(self) -> True:
        if self.num_timesteps % self.img_reconstruction_period == 0:
            vec_env = self.model.get_env()
            selected_idx = rd.randint(0, vec_env.num_envs - 1)
            sample_env = vec_env.envs[selected_idx].env

            run_episode_and_visualize_sampling(
                agent=self.model,
                env=sample_env,
                deterministic=self.deterministic,
                act_mae_model=self.act_mae_model,
                reconstruction_dir=self.reconstruction_dir,
                mask_sample=self.mask_sample,
                filename="ppo_agent",
                img_index=self.num_timesteps,
            )
            visualize_mae_reconstruction(
                sample_env.img.unsqueeze(0),
                self.mae_model,
                show=False,
                save_path=self.reconstruction_dir / f"mae_reconstruction_img={self.num_timesteps}"
            )

        return True


class ImgChangeCallback(BaseCallback):
    def __init__(self, img_change_period: int, img_dataloader: DataLoader):
        """Callback for updating the image of the exploration environment.

        Args:
            img_change_period (int): number of steps between env image update.
            img_dataloader (DataLoader): Dataloader of the env images.
        """
        super().__init__()
        self.img_change_period = img_change_period
        self.img_dataloader = img_dataloader
        self.img_iterator = iter(img_dataloader)

    def _on_step(self) -> True:
        if self.num_timesteps % self.img_change_period == 0:
            try:
                new_batch = next(self.img_iterator)
            except StopIteration:
                self.img_iterator = iter(self.img_dataloader)
                new_batch = next(self.img_iterator)

            vec_env = self.model.get_env()

            # Update each environment with its new image
            for j in range(new_batch.shape[0]):
                vec_env.env_method("set_img", new_batch[j], indices=j)

        return True


def train_ppo(params: dict, experiment_name: str = None, nested: bool = False) -> dict:

    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(nested=nested):

        # Experiment setup
        mlflow.transformers.autolog(disable=True)
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

        seed = params['seed'] if type(params['seed']) == int else params['seed'].item()
        mlflow.log_params(params)

        # Data
        torch_generator = torch.Generator().manual_seed(seed)
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)
        train_dataset = TinyImageNetDataset(split="train", transform=image_processor)
        val_dataset = TinyImageNetDataset(split="val", transform=image_processor)

        # Test with smaller datasets
        small_train_dataset = SmallImageDataset(train_dataset, 1)
        # small_val_dataset = SmallImageDataset(val_dataset, 1)
        small_val_dataset = small_train_dataset  # TODO: This is only for testing

        train_dataloader = DataLoader(small_train_dataset,
                                      batch_size=params['num_envs'],
                                      collate_fn=tiny_imagenet_collate_fn,
                                      generator=torch_generator,
                                      shuffle=True)

        val_dataloader = DataLoader(small_val_dataset,
                                    batch_size=params['num_envs'],
                                    collate_fn=tiny_imagenet_collate_fn,
                                    generator=torch_generator,
                                    shuffle=True)

        # Pretrained model and reward function
        mae_config = ViTMAEForPreTraining.config_class.from_pretrained("facebook/vit-mae-base")
        mae_config.seed = seed
        mae_config.patch_size = params['patch_size']
        mae_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", config=mae_config).to(DEVICE)

        act_mae_config = ActViTMAEForPreTraining.config_class.from_pretrained("facebook/vit-mae-base")
        act_mae_config.seed = seed
        act_mae_model = ActViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", config=act_mae_config).to(DEVICE)

        reward_function = RewardFunction(act_mae_model,
                                         num_samples=params['num_samples'],
                                         reward_increase=params['reward_increase'],
                                         patch_size=params['patch_size'],
                                         masking_ratio=params['masking_ratio'],
                                         mask_sample=params['mask_sample'],
                                         generator=torch_generator,
                                         )

        # Take one image as a dummy input for env initialization
        dummy_batch = next(iter(train_dataloader))
        env_config = ImageExplorationEnvConfig(steps_until_termination=params['steps_until_termination'],
                                               interval_reward_assignment=params['interval_reward_assignment'],
                                               sensor_size=params['sensor_size'],
                                               reward_function=reward_function,
                                               seed=seed
                                               )

        factory = ImageEnvFactory(dummy_batch, log_dir=str(logs_dir), env_config=env_config)
        vec_env = DummyVecEnv([factory(i) for i in range(params['num_envs'])])
        ppo_agent_policy_kwargs = dict(
            features_extractor_class=CustomResNetFeatureExtractor,
            features_extractor_kwargs=dict(resnet_features_dim=512, pos_features_dim=64),
            normalize_images=False
        )

        # Callbacks
        img_reconstruction_train_callback = ImgReconstructinoCallback(
            img_reconstruction_period=params['img_reconstruction_period'],
            mask_sample=params['mask_sample'],
            mae_model=mae_model,
            act_mae_model=act_mae_model,
            reconstruction_dir=train_img_reconstruction_dir,
            deterministic=False,
        )
        img_change_train_callback = ImgChangeCallback(
            img_change_period=params['img_change_period'],
            img_dataloader=train_dataloader,
        )
        callback_list_train = CallbackList([img_reconstruction_train_callback, img_change_train_callback])

        img_reconstruction_val_callback = ImgReconstructinoCallback(
            img_reconstruction_period=params['img_reconstruction_period'],
            mask_sample=params['mask_sample'],
            mae_model=mae_model,
            act_mae_model=act_mae_model,
            reconstruction_dir=train_img_reconstruction_dir,
            deterministic=True,
        )
        img_change_val_callback = ImgChangeCallback(
            img_change_period=params['img_change_period'],
            img_dataloader=val_dataloader,
        )
        callback_list_val = CallbackList([img_reconstruction_val_callback, img_change_val_callback])

        # PPO agent definition
        ppo_agent = PPO(
            params['policy'],
            vec_env,
            policy_kwargs=ppo_agent_policy_kwargs,
            device=params['device'],
            seed=seed,
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            verbose=1
        )
        ppo_agent_logger = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )
        ppo_agent.set_logger(ppo_agent_logger)

        # Training
        ppo_agent.learn(
            total_timesteps=params['total_timesteps'],
            callback=callback_list_train,
        )

        ppo_agent.save(models_dir / "ppo_model.zip")

        # Register the model in MLflow Model Registry
        model_uri = f"runs:/{run_id}/models"
        mlflow.register_model(model_uri, name=f"SB3_PPO_Model_{experiment_id}_{run_id}")

        # Regular MAE evaluation
        sum_loss = 0
        sum_reward = 0
        num_val_batches = len(val_dataloader)
        for batch in val_dataloader:
            with torch.no_grad():
                outputs = mae_model(batch)
            loss = outputs.loss
            reward = 1 / (1 + loss)

            sum_loss += loss
            sum_reward += reward

        mlflow.log_metric("eval/mae_loss", sum_loss / num_val_batches)
        mlflow.log_metric("eval/mae_reward", sum_reward / num_val_batches)

        # Reward MAE evaluation
        mean_reward, std_reward = evaluate_policy(
            ppo_agent,
            vec_env,
            n_eval_episodes=5,  # TODO: This should change according to val dataset length
            deterministic=True,
            return_episode_rewards=False,
            callback=callback_list_val
        )
        mlflow.log_metric("eval/mean_reward", mean_reward)
        mlflow.log_metric("eval/std_reward", std_reward)

        return {'loss': -mean_reward,
                'loss_variance': -std_reward,
                "status": STATUS_OK}


def objective(params: dict) -> dict:
    result = train_ppo(params, nested=True)

    return result


@app.command()
def ppo_fixed_params(experiment_name: str):
    train_ppo(PPO_PARAMS, experiment_name)


@app.command()
def ppo_fixed_params_seed_iter(experiment_name: str) -> None:
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://localhost:5000")
    seed_iter_param_space = PPO_PARAMS.copy()  # Fixed params, multiple seeds
    seed_iter_param_space['seed'] = hp.randint('seed', 100_000)

    with mlflow.start_run():
        # Conduct the hyperparameter search using Hyperopt
        trials = Trials()
        best = fmin(
            fn=objective,
            space=seed_iter_param_space,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials,
        )

        # Fetch the details of the best run
        best_run = sorted(trials.results, key=lambda x: x["loss"])[0]

        # Log the best parameters, loss, and model
        mlflow.log_params(best)
        mlflow.log_metric("eval/mean_reward", -best_run["loss"])
        mlflow.log_metric("eval/std_reward", -best_run["loss_variance"])


@app.command()
def ppo_hiperparameter_search(experiment_name: str) -> None:
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://localhost:5000")
    param_space = {
        'steps_until_termination': 16,
        'reward_increase': False,
        'num_samples': 1,
        'masking_ratio': 0.5,
        'sensor_size': 32,
        'patch_size': 16,
        'interval_reward_assignment': 16,
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)),
        'n_steps': 2048,
        'total_timesteps': 80_000,
        'batch_size': hp.choice('batch_size', [64, 128]),
        'n_epochs': 10,
        'clip_range': hp.uniform('clip_range', 0.2, 0.4),
        'gamma': 0.99,
        'gae_lambda': hp.uniform('gae_lambda', 0.8, 0.95),
        'ent_coef': hp.uniform('ent_coef', 0.01, 0.1),
        'policy': 'CnnPolicy',
        'vf_coef': 0.5,
        'device': DEVICE,
        'seed': 64533,
    }

    with mlflow.start_run():
        trials = Trials()
        best = fmin(
            fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
        )

        # Fetch the details of the best run
        best_run = sorted(trials.results, key=lambda x: x["loss"])[0]

        # Log the best parameters, loss, and model
        mlflow.log_params(best)
        mlflow.log_metric("eval/mean_reward", best_run["loss"])
        mlflow.log_metric("eval/std_reward", best_run["loss_variance"])


if __name__ == "__main__":
    # Remember to start server in cli from root dir: ```mlflow server --host 0.0.0.0 --port 5000```
    app()
