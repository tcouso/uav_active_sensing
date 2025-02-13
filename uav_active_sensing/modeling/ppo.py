# Original implementation can be bound at https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from uav_active_sensing.modeling.img_exploration_env import (
    ImageExplorationEnv,
    RewardFunction,
    ImageExplorationEnvConfig,
)
from uav_active_sensing.config import REPORTS_DIR, MODELS_DIR, DEVICE, IMG_BATCH_SIZE, SEED


@dataclass
class PPOConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = SEED
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = DEVICE == "cuda"
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "ImageExploration-v0"
    """the id of the environment"""
    total_timesteps: int = 10 * 128
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = IMG_BATCH_SIZE
    """the number of parallel game environments (equal to batch size)"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(img: torch.Tensor, reward_function: RewardFunction, gamma: float):
    def thunk():
        config = ImageExplorationEnvConfig(img=img, reward_function=reward_function)
        env = ImageExplorationEnv(config)  # TODO: make with seed
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def train_ppo(batch: torch.Tensor, reward_function: RewardFunction):
    ppo_config = PPOConfig()
    ppo_config.batch_size = int(ppo_config.num_envs * ppo_config.num_steps)
    ppo_config.minibatch_size = int(ppo_config.batch_size // ppo_config.num_minibatches)
    ppo_config.num_iterations = ppo_config.total_timesteps // ppo_config.batch_size
    run_name = f"{ppo_config.env_id}__{ppo_config.exp_name}__{ppo_config.seed}__{int(time.time())}"
    writer = SummaryWriter(f"{REPORTS_DIR}/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(ppo_config).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(ppo_config.seed)
    np.random.seed(ppo_config.seed)
    torch.manual_seed(ppo_config.seed)
    torch.backends.cudnn.deterministic = ppo_config.torch_deterministic

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(img.unsqueeze(0), reward_function, ppo_config.gamma) for img in batch]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=ppo_config.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (ppo_config.num_steps, ppo_config.num_envs) + envs.single_observation_space.shape
    ).to(DEVICE)
    actions = torch.zeros(
        (ppo_config.num_steps, ppo_config.num_envs) + envs.single_action_space.shape
    ).to(DEVICE)
    logprobs = torch.zeros((ppo_config.num_steps, ppo_config.num_envs)).to(DEVICE)
    rewards = torch.zeros((ppo_config.num_steps, ppo_config.num_envs)).to(DEVICE)
    dones = torch.zeros((ppo_config.num_steps, ppo_config.num_envs)).to(DEVICE)
    values = torch.zeros((ppo_config.num_steps, ppo_config.num_envs)).to(DEVICE)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=ppo_config.seed)
    next_obs = torch.Tensor(next_obs).to(DEVICE)
    next_done = torch.zeros(ppo_config.num_envs).to(DEVICE)

    print("Num iters: ", ppo_config.num_iterations)

    for iteration in range(1, ppo_config.num_iterations + 1):
        print(f"Curr iteration {iteration}")
        # Annealing the rate if instructed to do so.
        if ppo_config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / ppo_config.num_iterations
            lrnow = frac * ppo_config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, ppo_config.num_steps):
            # print(f"Curr step {step}")
            global_step += ppo_config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                action = torch.round(action).int()
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(DEVICE).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(DEVICE), torch.Tensor(next_done).to(
                DEVICE
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(DEVICE)
            lastgaelam = 0
            for t in reversed(range(ppo_config.num_steps)):
                if t == ppo_config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + ppo_config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + ppo_config.gamma * ppo_config.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(ppo_config.batch_size)
        clipfracs = []
        for epoch in range(ppo_config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, ppo_config.batch_size, ppo_config.minibatch_size):
                end = start + ppo_config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > ppo_config.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if ppo_config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - ppo_config.clip_coef, 1 + ppo_config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if ppo_config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -ppo_config.clip_coef,
                        ppo_config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ppo_config.ent_coef * entropy_loss + v_loss * ppo_config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo_config.max_grad_norm)
                optimizer.step()

            if ppo_config.target_kl is not None and approx_kl > ppo_config.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if ppo_config.save_model:
        model_path = Path(MODELS_DIR) / "runs" / run_name / f"{ppo_config.exp_name}.cleanrl_model"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
