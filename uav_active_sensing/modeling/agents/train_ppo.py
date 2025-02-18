import typer
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from stable_baselines3 import PPO

from uav_active_sensing.pytorch_datasets import TinyImageNetDataset, tiny_imagenet_collate_fn
from uav_active_sensing.modeling.img_env.img_exploration_env import RewardFunction, ImageExplorationEnv, ImageExplorationEnvConfig
from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.modeling.agents.rl_agent_feature_extractor import CustomResNetFeatureExtractor
from uav_active_sensing.config import DEVICE

# TODO: Define PPO config here


app = typer.Typer()


@app.command()
def train_ppo(dataset_path: Path = None, model_path: Path = None, img_processor_path: Path = None):

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)  # TODO: Download this in advance
    tiny_imagenet_train_dataset = TinyImageNetDataset(split="train", transform=image_processor)
    tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=1, collate_fn=tiny_imagenet_collate_fn)

    # Pretrained model and reward function
    model = ActViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")  # TODO: Download this in advance
    reward_function = RewardFunction(model)

    # Create a dummy environment to initialize the model
    dummy_batch = next(iter(tiny_imagenet_train_loader))  # Take one image as a dummy input for env initialization
    env_config = ImageExplorationEnvConfig(img=dummy_batch, reward_function=reward_function)
    env = ImageExplorationEnv(env_config)

    ppo_agent_policy_kwargs = dict(
        features_extractor_class=CustomResNetFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # TODO: Set these params to sensible values
    # TODO: Implement some experiment monitoring logic (tensorboard, or something of the sort)

    rl_num_envs = 1
    rl_batch_size = 4
    rl_num_steps = rl_batch_size * rl_num_envs * 4

    ppo_agent = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=ppo_agent_policy_kwargs,
        verbose=1,
        device=DEVICE,
        n_steps=rl_num_steps,
        batch_size=rl_batch_size
    )

    for i, batch in enumerate(tiny_imagenet_train_loader):
        vec_env = ppo_agent.get_env()
        vec_env.env_method("set_img", batch)
        ppo_agent.learn(total_timesteps=2 * rl_num_steps, progress_bar=False)

        # TODO: One grad update per batch


if __name__ == "__main__":
    app()
