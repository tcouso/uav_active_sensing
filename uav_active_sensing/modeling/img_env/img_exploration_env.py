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


# def make_kernel_size_odd(n: int) -> int:
#     assert n > 0

#     return n if n % 2 == 1 else n - 1


def make_kernel_size_odd(t: torch.Tensor) -> torch.Tensor:
    # assert torch.all(t > 0), "All values must be greater than 0"

    return torch.where(abs(t % 2) == 1, t, t - 1)


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
        # self.__sensor_pos = [rd.randint(0, self.img_height), rd.randint(0, self.img_width)]
        self.__sensor_pos = torch.stack([
            torch.randint(0, self.img_height, (self.batch_size,), dtype=torch.int32),
            torch.randint(0, self.img_width, (self.batch_size,), dtype=torch.int32)
        ], dim=1)

        self._min_kernel_size = torch.ones(self.batch_size, dtype=torch.int32)
        self._max_kernel_size = torch.full((self.batch_size, 1), self.img_height - self.sensor_height + 1, dtype=torch.int32)
        self._max_kernel_size = make_kernel_size_odd(self._max_kernel_size)
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

    def _get_info(self):
        info = {}

        return info

    # TODO: Adjust this to batch logic

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

    # TODO: Adjust this to batch logic
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

    # # TODO: Adjust this to batch logic
    # @property
    # def fov_bbox(self) -> List[int]:
    #     """
    #     Computes the bounding box of the field of view (FoV) based on the current sensor position and kernel size.

    #     Returns:
    #         List[int]: A list of coordinates [top, bottom, left, right] defining the FoV bounding box.
    #     """
    #     fov_height = int(self.sensor_height + self._kernel_size - 1)
    #     fov_width = int(self.sensor_width + self._kernel_size - 1)

    #     top = self._sensor_pos[:, 0]
    #     left = self._sensor_pos[:, 1]
    #     bottom = self._sensor_pos[:, 0] + fov_height
    #     right = self._sensor_pos[:, 1] + fov_width

    #     return torch.cat((top, bottom, left, right), dim=1)

    # @property
    # def fov_bbox(self) -> torch.Tensor:
    #     """Retuns fov positions in top left bottom right format stacked over batch"""

    #     fov_size = self._kernel_size - 1 # TODO: Adapt to batched kernel sizes

    #     offset = torch.tensor([fov_size, fov_size], device=self.device)
    #     bottom_right = self._sensor_pos + offset
    #     fov_bbox = torch.cat((self._sensor_pos, bottom_right), dim=1)

    #     return fov_bbox

    @property
    def fov_bbox(self) -> torch.Tensor:
        """Returns fov positions in top left bottom right format stacked over batch"""

        fov_size = self._kernel_size - 1  # TODO: kernel size should be positive
        # print(fov_size)
        offset = torch.cat((fov_size.reshape(-1, 1), fov_size.reshape(-1, 1)), dim=1).to(self.device)

        # print(offset)
        # Calculate the bottom-right corner for each image in the batch
        bottom_right = self._sensor_pos + offset  # Broadcasting to match batch size

        # Concatenate top-left and bottom-right corners along dimension 1
        fov_bbox = torch.cat((self._sensor_pos, bottom_right), dim=1)  # Shape: (batch_size, 4)

        return fov_bbox

    # TODO: Adjust this to batch logic

    @property
    def _sensor_max_pos(self) -> torch.Tensor:
        """
        Computes the maximum valid position for the sensor based on the current FoV.

        """
        fov_bbox = self.fov_bbox

        assert fov_bbox.shape == (self.batch_size, 4)

        fov_height = fov_bbox[:, 0] - fov_bbox[:, 0]
        fov_width = fov_bbox[:, 3] - fov_bbox[:, 1]

        sensor_max_height = self.img_height - fov_height
        sensor_max_width = self.img_width - fov_width

        return torch.cat((sensor_max_height.reshape(-1, 1), sensor_max_width.reshape(-1, 1)), dim=1)

    # TODO: Adjust this to batch logic
    @property
    def _sensor_pos(self) -> torch.Tensor:

        return self.__sensor_pos

    # TODO: Adjust this to batch logic
    @_sensor_pos.setter
    def _sensor_pos(self, new_position: torch.Tensor) -> None:
        """
        Sets the sensor position, ensuring it stays within valid bounds.
        """
        assert new_position.shape == self.__sensor_pos.shape

        self.__sensor_pos = torch.clamp(new_position, min=self._sensor_min_pos, max=self._sensor_max_pos)

    # TODO: Adjust this to batch logic
    @property
    def _kernel_size(self) -> torch.Tensor:

        return self.__kernel_size

    # TODO: Adjust this to batch logic
    @_kernel_size.setter
    def _kernel_size(self, new_kernel_size: torch.Tensor) -> None:
        max_kernel_size_from_sensor_pos = min(self.img_height, self.img_width) - torch.amax(self._sensor_pos, dim=1)  # Current position restricts kernel size
        max_kernel_size_from_sensor_pos = make_kernel_size_odd(max_kernel_size_from_sensor_pos)

        new_kernel_size = torch.clamp(new_kernel_size, min=self._min_kernel_size, max=max_kernel_size_from_sensor_pos)
        self.__kernel_size = make_kernel_size_odd(new_kernel_size)

    # TODO: Adjust this to batch logic
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

    # TODO: Adjust this to batch logic
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

    # TODO: Adjust this to batch logic
    def _update_sampled_img(self) -> None:
        """
        Updates the sampled image based on the current sensor position and kernel size.
        """
        fov_bbox = self.fov_bbox

        top = fov_bbox[:, 0].flatten()
        left = fov_bbox[:, 1].flatten()
        bottom = fov_bbox[:, 2].flatten()
        right = fov_bbox[:, 3].flatten()

        print(top.shape)

        batch_indices = torch.arange(self.batch_size, device=self.device)[:, None, None]  # Shape: (B, 1, 1)

        # Perform batch-wise indexing # TODO: Handle different size accross diferent images of batch
        # there might be a breaking point here

        obs = self.img[
            batch_indices,  # Selects each batch index
            :,              # Selects all channels
            top[:, None, None]:bottom[:, None, None],  # Row range for each image
            left[:, None]:right[:, None]  # Column range for each image
        ].clone()

        # top, bottom, left, right = self.fov_bbox
        # obs = self.img[:, :, top:bottom, left:right].clone()

        if self._kernel_size > self._min_kernel_size:
            obs = self._apply_blur(obs)
            obs = obs

        # obs = self._filter_high_blur_obs(obs) # TODO: Adapt to batch logic

        print(obs)
        print(obs.shape)

        # self.sampled_img[:, :, top:bottom, left:right] = obs

    # TODO: Adjust this to batch logic
    def move(self, action: torch.Tensor) -> None:
        """
        Moves the sensor and updates the observation.

        Args:
            dx (int): The change in the x-direction.
            dy (int): The change in the y-direction.
            dz (int): The change in the kernel size.
        """

        assert action.shape == (self.batch_size, 3)

        self._sensor_pos += action[:, :2]
        self._kernel_size += action[:, 2]
        self._update_sampled_img()
