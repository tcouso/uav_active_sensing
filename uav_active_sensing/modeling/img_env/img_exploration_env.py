import random as rd
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.config import DEVICE, set_seed, SEED

set_seed()

# TODO: Guaratee deterministic behaviour given seed
# TODO: Handle batch envs


def make_kernel_size_odd(n: int) -> int:
    assert n > 0

    return n if n % 2 == 1 else n - 1


class RewardFunction:
    def __init__(self, model):
        self.model: ActViTMAEForPreTraining = model

    def __call__(self, img: torch.Tensor, sampled_img: torch.Tensor) -> float:

        with torch.no_grad():
            outputs = self.model(img, sampled_img)
        loss = outputs.loss
        reward = 1 / (1 + loss)
        reward = reward.detach().item()

        return reward


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

        self._sensor_min_pos = [0, 0]
        self.__sensor_pos = [rd.randint(0, self.img_height), rd.randint(0, self.img_width)]

        self._min_kernel_size = 1
        self._max_kernel_size = make_kernel_size_odd(self.img_height - self.sensor_height + 1)
        self.__kernel_size = self._min_kernel_size

        self.v_max_x = env_config.v_max_x
        self.v_max_y = env_config.v_max_y
        self.v_max_z = env_config.v_max_z

        self._sampled_kernel_size_mask = torch.full_like(
            self.img, fill_value=self._max_kernel_size, dtype=torch.int32
        )
        self.sampled_img = torch.full_like(self.img, float("nan"), device=self.device)

        # Gymnasium interface
        self._max_steps = env_config.max_steps
        self._step_count = 0

        self.reward_function = env_config.reward_function
        self.interval_reward_assignment = env_config.interval_reward_assignment

        self.observation_space = spaces.Box(
            low=-3.0,  # Bounds are a conservative estimate based on ImageNet normalization params
            high=3.0,
            shape=(1, 3, 224, 224),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array(
                [-1, -1, -1], dtype=np.float32
            ),
            high=np.array(
                [1, 1, 1], dtype=np.float32
            ),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:

        obs = torch.nan_to_num(self.sampled_img, nan=0.0).detach().numpy()

        return obs

    def _get_info(self):
        info = {}

        return info

    def _denormalize_action(self, action: Tuple[float, float, float]) -> Tuple[int, int, int]:
        # Env action space bounds
        low = np.array([-self.v_max_x, -self.v_max_y, -self.v_max_z], dtype=np.float32)
        high = np.array([self.v_max_x, self.v_max_y, self.v_max_z], dtype=np.float32)

        # Denormalize action
        denormalized_action = low + (action + 1) * 0.5 * (high - low)

        dx, dy, dz = denormalized_action.astype(int)  # Enforce int actions in this env

        return dx, dy, dz

    def reset(self, seed: Optional[int] = SEED, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed != None:
            super().reset(seed=seed)

        # Clear episode variables
        self.sampled_img = torch.full_like(self.img, float("nan"), device=self.device)
        self._sampled_kernel_size_mask = torch.full_like(
            self.img, fill_value=self._max_kernel_size, dtype=torch.int32
        )
        self._step_count = 0

        action = self.action_space.sample()
        dx, dy, dz = self._denormalize_action(action)
        self.move(dx, dy, dz)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool, bool, dict]:

        dx, dy, dz = self._denormalize_action(action)
        self.move(dx, dy, dz)
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
    def fov_bbox(self) -> List[int]:
        """
        Computes the bounding box of the field of view (FoV) based on the current sensor position and kernel size.

        Returns:
            List[int]: A list of coordinates [top, bottom, left, right] defining the FoV bounding box.
        """
        fov_height = int(self.sensor_height + self._kernel_size - 1)
        fov_width = int(self.sensor_width + self._kernel_size - 1)

        top = self._sensor_pos[0]
        left = self._sensor_pos[1]
        bottom = self._sensor_pos[0] + fov_height
        right = self._sensor_pos[1] + fov_width

        return [top, bottom, left, right]

    @property
    def _sensor_max_pos(self) -> Tuple[int, int]:
        """
        Computes the maximum valid position for the sensor based on the current FoV.

        Returns:
            Tuple[int, int]: The maximum (y, x) position the sensor can occupy.
        """
        top, bottom, left, right = self.fov_bbox

        fov_height = bottom - top
        fov_width = right - left
        sensor_max_height = self.img_height - fov_height
        sensor_max_width = self.img_width - fov_width

        return (sensor_max_height, sensor_max_width)

    @property
    def _sensor_pos(self) -> Tuple[int, int]:
        """
        Gets the current sensor position.

        Returns:
            Tuple[int, int]: The current (y, x) position of the sensor.
        """
        return tuple(self.__sensor_pos)

    @_sensor_pos.setter
    def _sensor_pos(self, new_position: Tuple[int, int]) -> None:
        """
        Sets the sensor position, ensuring it stays within valid bounds.

        Args:
            new_position (Tuple[int, int]): The new (y, x) position for the sensor.
        """
        x, y = new_position
        # assert (x.is_integer() and y.is_integer())
        self.__sensor_pos[0] = max(min(x, self._sensor_max_pos[0]), self._sensor_min_pos[0])
        self.__sensor_pos[1] = max(min(y, self._sensor_max_pos[1]), self._sensor_min_pos[1])

    @property
    def _kernel_size(self) -> int:
        """
        Gets the current kernel size.

        Returns:
            int: The current kernel size.
        """
        return self.__kernel_size

    @_kernel_size.setter
    def _kernel_size(self, new_kernel_size: int) -> None:
        """
        Sets the kernel size, ensuring it stays within valid bounds.

        Args:
            kernel_size (int): The new kernel size.
        """
        max_kernel_size_from_sensor_pos = make_kernel_size_odd(
            min(self.img_height, self.img_width) - max(self._sensor_pos)
        )  # Current position restricts kernel size
        new_kernel_size = max(
            min(new_kernel_size, max_kernel_size_from_sensor_pos), self._min_kernel_size
        )

        self.__kernel_size = make_kernel_size_odd(new_kernel_size)

    def _apply_blur(self, window: torch.Tensor) -> torch.Tensor:
        """
        Applies an averaging blur to the input window tensor while considering margin artifacts.

        Args:
            window (torch.Tensor): A tensor of shape (C, H, W), where C is the number of channels,
                                H is the height, and W is the width of the image.

        Returns:
            torch.Tensor: A blurred tensor with the same shape as the input window.
        """
        padding = int(self._kernel_size // 2)
        window_padded = F.pad(window, (padding, padding, padding, padding), mode="reflect")
        blurred = F.avg_pool2d(window_padded, kernel_size=self._kernel_size, stride=1, padding=0)

        # assert blurred.shape == window.shape

        return blurred

    def _filter_high_blur_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Filters observations to ensure higher blur levels do not overwrite lower ones.

        Args:
            obs (torch.Tensor): The observed tensor.

        Returns:
            torch.Tensor: The filtered observation tensor.
        """
        top, bottom, left, right = self.fov_bbox

        prev_mask = self._sampled_kernel_size_mask[:, :, top:bottom, left:right]
        curr_mask = torch.full_like(prev_mask, fill_value=self._kernel_size)

        # Blur mask update
        updated_mask = curr_mask < prev_mask
        self._sampled_kernel_size_mask[:, :, top:bottom, left:right][updated_mask] = curr_mask[
            updated_mask
        ]

        # Observation update
        prev_obs = self.sampled_img[:, :, top:bottom, left:right]
        obs_to_update = curr_mask > prev_mask
        obs[obs_to_update] = prev_obs[obs_to_update]

        return obs

    def _update_sampled_img(self) -> None:
        """
        Updates the sampled image based on the current sensor position and kernel size.
        """
        top, bottom, left, right = self.fov_bbox
        obs = self.img[:, :, top:bottom, left:right].clone()

        if self._kernel_size > self._min_kernel_size:
            obs = self._apply_blur(obs)
            obs = obs

        obs = self._filter_high_blur_obs(obs)

        self.sampled_img[:, :, top:bottom, left:right] = obs

    def move(self, dx: int, dy: int, dz: int) -> None:
        """
        Moves the sensor and updates the observation.

        Args:
            dx (int): The change in the x-direction.
            dy (int): The change in the y-direction.
            dz (int): The change in the kernel size.
        """
        # assert (dx.is_integer() and dy.is_integer() and dz.is_integer())

        self._sensor_pos = (
            self._sensor_pos[0] + dy,
            self._sensor_pos[1] + dx,
        )
        self._kernel_size += dz
        self._update_sampled_img()
