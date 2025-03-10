from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.config import DEVICE

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
    seed: int = None
    img_batch_size: int = 1
    patch_size: int = 2 * 16
    steps_until_termination: int = 30
    interval_reward_assignment: int = 5
    v_max_x: int = patch_size
    v_max_y: int = patch_size
    v_max_z: int = 0

    # Set during execution
    img_sensor_ratio: float = None
    img: torch.Tensor = None
    reward_function: RewardFunction = None

# TODO: Ensure cuda placing


class ImageExplorationEnv(gym.Env):

    def __init__(self, env_config: ImageExplorationEnvConfig) -> None:
        super().__init__()
        self.device = env_config.device
        self.seed = env_config.seed
        self.generator = torch.Generator().manual_seed(self.seed)
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
            torch.randint(0, self.img_height - self.sensor_height, (self.batch_size,), dtype=torch.int32, generator=self.generator),
            torch.randint(0, self.img_width - self.sensor_width, (self.batch_size,), dtype=torch.int32, generator=self.generator)
        ], dim=1)
        # max_h = (self.img_height - self.sensor_height) // self.sensor_height + 1
        # max_w = (self.img_width - self.sensor_width) // self.sensor_width + 1

        # self.__sensor_pos = torch.stack([
        #     torch.randint(0, max_h, (self.batch_size,), dtype=torch.int32, generator=self.generator) * self.sensor_height,
        #     torch.randint(0, max_w, (self.batch_size,), dtype=torch.int32, generator=self.generator) * self.sensor_width,
        # ], dim=1)

        self._min_kernel_size = torch.ones(self.batch_size, dtype=torch.int32)
        self.__kernel_size = torch.ones(self.batch_size, dtype=torch.int32)

        self.v_max_x = env_config.v_max_x
        self.v_max_y = env_config.v_max_y
        self.v_max_z = env_config.v_max_z

        self._sampled_kernel_size_mask = torch.full_like(self.img, fill_value=self.img_height // self.sensor_height, dtype=torch.int32)  # Max possible kernel size
        self.sampled_img = torch.full_like(self.img, float("nan"), device=self.device)

        self.max_steps = env_config.steps_until_termination
        self._step_count = 0

        self._reward_function = env_config.reward_function
        self._interval_reward_assignment = env_config.interval_reward_assignment

        self.observation_space = spaces.Box(
            low=-3.0,  # Bounds are a conservative estimate based on ImageNet normalization params
            high=3.0,
            shape=(self.batch_size, 3, 224, 224),
            dtype=np.float32,
            seed=self.seed
        )

        self.action_space = spaces.Box(
            low=np.tile(np.array([-1, -1, -1], dtype=np.float32), (self.batch_size, 1)),
            high=np.tile(np.array([1, 1, 1], dtype=np.float32), (self.batch_size, 1)),
            dtype=np.float32,
            seed=self.seed
        )

    def _get_obs(self) -> np.ndarray:

        obs = torch.nan_to_num(self.sampled_img, nan=0.0).cpu().numpy()

        return obs

    def _get_info(self):  # TODO: Is there any info that I should be giving?
        info = {}

        return info

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        # Env action space bounds
        low = torch.tensor([-self.v_max_x, -self.v_max_y, -self.v_max_z], dtype=torch.float32, device=action.device)
        high = torch.tensor([self.v_max_x, self.v_max_y, self.v_max_z], dtype=torch.float32, device=action.device)

        # Denormalize action
        denormalized_action = low + (action + 1) * 0.5 * (high - low)

        return denormalized_action.to(dtype=torch.int)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed != None:
            super().reset(seed=seed)

        self.__sensor_pos = torch.stack([
            torch.randint(0, self.img_height - self.sensor_height, (self.batch_size,), dtype=torch.int32, generator=self.generator),
            torch.randint(0, self.img_width - self.sensor_width, (self.batch_size,), dtype=torch.int32, generator=self.generator)
        ], dim=1)
        self.sampled_img = torch.full_like(self.img, float("nan"), device=self.device)
        self._sampled_kernel_size_mask = torch.full_like(
            self.img,
            fill_value=self.img_height // self.sensor_height,  # Max possible kernel size
            dtype=torch.int32
        )
        self._kernel_size = torch.ones(self.batch_size, dtype=torch.int32)
        self._step_count = 0

        self._update_sampled_img()  # Initial env observation is without movement
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray, eval: bool=False) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # print(f"Action pre norm: {action}")
        action = torch.from_numpy(action)
        action = self._denormalize_action(action)
        # print(f"Action post norm: {action}")

        self.move(action)
        observation = self._get_obs()

        if (self._step_count % self._interval_reward_assignment == 0) and not eval:
            reward = self._reward_function(self.img, self.sampled_img)

        else:
            reward = float(0)

        terminated = self._step_count >= self.max_steps
        self._step_count += 1

        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def set_img(self, new_img: torch.Tensor) -> None:
        """Exposed setter method for updating images between batches"""
        self.img = new_img

    @property
    def step_count(self):

        return self._step_count

    @property
    def fov_bbox(self) -> torch.Tensor:
        """Returns fov positions in top left bottom right format stacked over batch. Depends on current kernel size"""

        offset = torch.stack((self.sensor_height * self._kernel_size, self.sensor_width * self._kernel_size), dim=1)
        bottom_right = self._sensor_pos + offset

        fov_bbox = torch.cat((self._sensor_pos, bottom_right), dim=1)

        return fov_bbox

    @property
    def sensor_max_pos_from_kernel_size(self) -> torch.Tensor:
        """
        Computes the maximum valid position for the sensor based on the current FoV (dependent on kernel size).

        """
        fov_bbox = self.fov_bbox

        fov_height = fov_bbox[:, 2] - fov_bbox[:, 0]
        fov_width = fov_bbox[:, 3] - fov_bbox[:, 1]

        # print(f"FoV height: {fov_height}")
        # print(f"FoV width: {fov_width}")

        sensor_max_height_pos = self.img_height - fov_height
        sensor_max_width_pos = self.img_width - fov_width
        # print(f"sensor_max_height_pos: {sensor_max_height_pos}")
        # print(f"sensor_max_width_pos: {sensor_max_width_pos}")
        sensor_max_pos = torch.tensor([[self.img_height - self.sensor_height, self.img_width - self.sensor_width]] * self.batch_size, dtype=torch.int32)

        sensor_pos = torch.cat((sensor_max_height_pos.reshape(-1, 1), sensor_max_width_pos.reshape(-1, 1)), dim=1)
        sensor_pos = torch.clamp(sensor_pos, min=self._sensor_min_pos, max=sensor_max_pos)

        assert torch.all(sensor_pos[:, 0] <= self.img_height), "Sensor max height can't be greater than img height"
        assert torch.all(sensor_pos[:, 1] <= self.img_width), "Sensor max width can't be greater than img width"
        assert torch.all(sensor_pos >= self._sensor_min_pos), "Sensor max position can't be less than min position"

        return sensor_pos

    @property
    def _sensor_pos(self) -> torch.Tensor:

        return self.__sensor_pos

    @_sensor_pos.setter
    def _sensor_pos(self, new_position: torch.Tensor) -> None:
        """
        Sets the sensor position, ensuring it stays within valid bounds.
        """

        # print(f"New pos before clamping: {new_position}")
        new_position = torch.clamp(new_position, min=self._sensor_min_pos, max=self.sensor_max_pos_from_kernel_size)
        # print(f"New pos after clamping: {new_position}")

        self.__sensor_pos = new_position

    @property
    def max_kernel_size_from_sensor_pos(self) -> torch.Tensor:
        max_kernel_h_from_pos = (self.img_height - self._sensor_pos[:, 0]) // self.sensor_height
        max_kernel_w_from_pos = (self.img_width - self._sensor_pos[:, 1]) // self.sensor_width
        kernel_size = torch.minimum(max_kernel_h_from_pos, max_kernel_w_from_pos)
        kernel_size = make_kernel_size_odd(kernel_size)

        return kernel_size

    @property
    def _kernel_size(self) -> torch.Tensor:

        return self.__kernel_size

    @_kernel_size.setter
    def _kernel_size(self, new_kernel_size: torch.Tensor) -> None:
        # print(f"Previous kernel size: {self._kernel_size}")

        new_kernel_size = torch.clamp(new_kernel_size, min=self._min_kernel_size, max=self.max_kernel_size_from_sensor_pos)
        new_kernel_size = make_kernel_size_odd(new_kernel_size)
        # print(f"New kernel size: {new_kernel_size}")

        assert torch.all(new_kernel_size > 0), "All values of new kernel size must be greater than 0"

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

        if window.shape[1] == 0 or window.shape[2] == 0:
            raise ValueError(f"Invalid window shape: {window.shape}")

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

        # assert action.shape == (self.batch_size, 3), f"Wrong shape. Expected {(self.batch_size, 3)}, Actual: {action.shape}"

        self._sensor_pos += action[:, :2]
        self._kernel_size += action[:, 2]
        self._update_sampled_img()
