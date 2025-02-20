import random as rd
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import math
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.config import DEVICE, set_seed, SEED

set_seed()

# TODO: Guaratee deterministic behaviour given seed


def make_kernel_size_odd(t: torch.Tensor) -> torch.Tensor:

    return torch.where(abs(t % 2) == 1, t, t - 1)


class RewardFunction:
    def __init__(self, model):
        self.model: ActViTMAEForPreTraining = model

    def __call__(self, img: torch.Tensor, sampled_img: torch.Tensor) -> float:

        batch_size = img.shape[0]
        batch_reward = torch.zeros(batch_size, dtype=torch.float32)

        for img_i in range(batch_size):
            with torch.no_grad():
                outputs = self.model(img[img_i].unsqueeze(0), sampled_img[img_i].unsqueeze(0))
            loss = outputs.loss
            reward_i = 1 / (1 + loss)
            batch_reward[img_i] = reward_i

        return batch_reward.sum().item()


@dataclass
class ImageExplorationEnvConfig:
    device: str = DEVICE
    patch_size: int = 16
    max_steps: int = 30
    interval_reward_assignment: int = 2
    v_max_x: int = 16
    v_max_y: int = 16
    v_max_z: int = 16

    # Set during execution
    img_sensor_ratio: float = None
    img: torch.Tensor = None
    reward_function: RewardFunction = None

# TODO: Ensure cuda placing
class ImageExplorationEnv(gym.Env):

    def __init__(self, env_config: ImageExplorationEnvConfig) -> None:
        super().__init__()
        self.device = env_config.device
        self.img = env_config.img
        self.img_height, self.img_width = self.img.shape[2:]
        self.batch_size = self.img.shape[0]

        if env_config.img_sensor_ratio is not None:
            self.sensor_height = self.img_height // env_config.img_sensor_ratio
            self.sensor_width = self.img_width // env_config.img_sensor_ratio
        else:
            self.sensor_height = env_config.patch_size
            self.sensor_width = env_config.patch_size

        self._sensor_min_pos = torch.zeros((self.batch_size, 2), dtype=torch.int32)
        self.__sensor_pos = torch.stack([
            torch.randint(0, self.img_height, (self.batch_size,), dtype=torch.int32),
            torch.randint(0, self.img_width, (self.batch_size,), dtype=torch.int32)
        ], dim=1)

        self._min_kernel_size = torch.ones(self.batch_size, dtype=torch.int32)
        self.__kernel_size = torch.ones(self.batch_size, dtype=torch.int32)

        self.v_max_x = env_config.v_max_x
        self.v_max_y = env_config.v_max_y
        self.v_max_z = env_config.v_max_z

        self._sampled_kernel_size_mask = torch.full_like(self.img, fill_value=float('inf'), dtype=torch.float32)

        self.sampled_img = torch.full_like(self.img, float("nan"), device=self.device)

        # Gymnasium interface
        self._max_steps = env_config.max_steps
        self._step_count = 0

        self.reward_function = env_config.reward_function
        self.interval_reward_assignment = env_config.interval_reward_assignment

        self.observation_space = spaces.Box(
            low=-3.0,  # Bounds are a conservative estimate based on ImageNet normalization params
            high=3.0,
            shape=(self.batch_size, 3, 224, 224),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.tile(np.array([-1, -1, -1], dtype=np.float32), (self.batch_size, 1)),
            high=np.tile(np.array([1, 1, 1], dtype=np.float32), (self.batch_size, 1)),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:

        obs = torch.nan_to_num(self.sampled_img, nan=0.0).detach().numpy()

        return obs

    def _get_info(self):  # TODO: Look references of good info to give
        info = {}

        return info

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        # Env action space bounds
        low = torch.tensor([-self.v_max_x, -self.v_max_y, -self.v_max_z], dtype=torch.float32, device=action.device)
        high = torch.tensor([self.v_max_x, self.v_max_y, self.v_max_z], dtype=torch.float32, device=action.device)

        # Denormalize action
        denormalized_action = low + (action + 1) * 0.5 * (high - low)

        return denormalized_action.to(dtype=torch.int)

    def reset(self, seed: Optional[int] = SEED, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed != None:
            super().reset(seed=seed)

        # Clear episode variables
        self.sampled_img = torch.full_like(self.img, float("nan"), device=self.device)
        max_kernel_size = self.img_height - self.sensor_height + 1
        self._sampled_kernel_size_mask = torch.full_like(
            self.img,
            fill_value=max_kernel_size,
            dtype=torch.int32
        )
        self._step_count = 0

        action = self.action_space.sample()
        action = torch.from_numpy(action)
        action = self._denormalize_action(action)
        self.move(action)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:

        action = torch.from_numpy(action)
        action = self._denormalize_action(action)
        self.move(action)
        observation = self._get_obs()

        if self._step_count % self.interval_reward_assignment == 0:
            reward = self.reward_function(self.img, self.sampled_img)

        else:
            reward = float(0)

        terminated = self._step_count >= self._max_steps
        self._step_count += 1

        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def set_img(self, new_img: torch.Tensor) -> None:

        self.img = new_img

    @property
    def step_count(self):

        return self._step_count

    @property
    def fov_bbox(self) -> torch.Tensor:
        """Returns fov positions in top left bottom right format stacked over batch"""

        fov_size = self._kernel_size - 1
        sensor_size = torch.tensor((self.sensor_height, self.sensor_width), dtype=torch.int32)
        offset = torch.cat((fov_size.reshape(-1, 1), fov_size.reshape(-1, 1)), dim=1).to(self.device)
        bottom_right = sensor_size + offset
        fov_bbox = torch.cat((self._sensor_pos, self._sensor_pos + bottom_right), dim=1)

        return fov_bbox

    @property
    def _sensor_max_pos(self) -> torch.Tensor:
        """
        Computes the maximum valid position for the sensor based on the current FoV.

        """
        fov_bbox = self.fov_bbox

        fov_height = fov_bbox[:, 2] - fov_bbox[:, 0]
        fov_width = fov_bbox[:, 3] - fov_bbox[:, 1]

        # print(f"FoV height: {fov_height}")
        # print(f"FoV width: {fov_width}")

        sensor_max_height = self.img_height - fov_height
        sensor_max_width = self.img_width - fov_width
        # print(f"sensor_max_height: {sensor_max_height}")
        # print(f"sensor_max_width: {sensor_max_width}")

        sensor_max_pos = torch.cat((sensor_max_height.reshape(-1, 1), sensor_max_width.reshape(-1, 1)), dim=1)

        # assert torch.all(sensor_max_pos[:, 0] <= self.img_height), "Sensor max height can't be greater than img height"
        # assert torch.all(sensor_max_pos[:, 1] <= self.img_width), "Sensor max width can't be greater than img width"

        return sensor_max_pos

    @property
    def _sensor_pos(self) -> torch.Tensor:

        return self.__sensor_pos

    @_sensor_pos.setter
    def _sensor_pos(self, new_position: torch.Tensor) -> None:
        """
        Sets the sensor position, ensuring it stays within valid bounds.
        """

        # assert torch.all(self._sensor_max_pos >= self._sensor_min_pos), "Sensor max position can't be less than min position"
        # print(f"New pos before clamping: {new_position}")
        new_position = torch.clamp(new_position, min=self._sensor_min_pos, max=self._sensor_max_pos)
        # print(f"New pos after clamping: {new_position}")

        self.__sensor_pos = new_position

    @property
    def _kernel_size(self) -> torch.Tensor:

        return self.__kernel_size

    @_kernel_size.setter
    def _kernel_size(self, new_kernel_size: torch.Tensor) -> None:
        # print(f"Previous kernel size: {self._kernel_size}")
        max_kernel_size_from_sensor_pos = torch.minimum(self.img_height - self._sensor_pos[:, 0], self.img_width - self._sensor_pos[:, 1])
        max_kernel_size_from_sensor_pos = make_kernel_size_odd(max_kernel_size_from_sensor_pos)

        new_kernel_size = torch.clamp(new_kernel_size, min=self._min_kernel_size, max=max_kernel_size_from_sensor_pos)
        new_kernel_size = make_kernel_size_odd(new_kernel_size)
        # print(f"New kernel size: {new_kernel_size}")

        # assert torch.all(new_kernel_size > 0), "All values of new kernel size must be greater than 0"

        self.__kernel_size = new_kernel_size

    @staticmethod
    def _apply_blur(window: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Applies an averaging blur to the input window tensor while considering margin artifacts.

        Args:
            window (torch.Tensor): A tensor of shape (C, H, W), where C is the number of channels,
                                H is the height, and W is the width of the image.

        Returns:
            torch.Tensor: A blurred tensor with the same shape as the input window.
        """
        padding = int(kernel_size // 2)
        window_padded = F.pad(window, (padding, padding, padding, padding), mode="reflect")
        blurred = F.avg_pool2d(window_padded, kernel_size=kernel_size, stride=1, padding=0)

        # # assert blurred.shape == window.shape

        return blurred

    def _update_sampled_img(self) -> None:
        """
        Updates the sampled image based on the current sensor position and kernel size.
        """
        fov_bbox = self.fov_bbox
        for img_i in range(self.batch_size):

            top = fov_bbox[img_i, 0].item()
            left = fov_bbox[img_i, 1].item()
            bottom = fov_bbox[img_i, 2].item()
            right = fov_bbox[img_i, 3].item()

            obs = self.img[img_i, :, top:bottom, left:right].clone()

            if self._kernel_size[img_i] > self._min_kernel_size[img_i]:
                obs = ImageExplorationEnv._apply_blur(obs, self._kernel_size[img_i].item())

            # Filter observations to ensure higher blur levels do not overwrite lower ones
            prev_mask = self._sampled_kernel_size_mask[img_i, :, top:bottom, left:right]
            curr_mask = torch.full_like(prev_mask, fill_value=self._kernel_size[img_i].item())

            # Blur mask update
            updated_mask = curr_mask < prev_mask
            self._sampled_kernel_size_mask[img_i, :, top:bottom, left:right][updated_mask] = curr_mask[
                updated_mask
            ]

            # Observation update
            prev_obs = self.sampled_img[img_i, :, top:bottom, left:right]
            obs_to_update = curr_mask > prev_mask
            obs[obs_to_update] = prev_obs[obs_to_update]

            self.sampled_img[img_i, :, top:bottom, left:right] = obs

    def move(self, action: torch.Tensor) -> None:
        """
        Moves the sensor and updates the observation.

        Args:
            dx (int): The change in the x-direction.
            dy (int): The change in the y-direction.
            dz (int): The change in the kernel size.
        """

        # assert action.shape == (self.batch_size, 3)

        self._sensor_pos += action[:, :2]
        self._kernel_size += action[:, 2]
        self._update_sampled_img()
