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
    def __init__(self, model: ActViTMAEForPreTraining,
                 num_samples: int, patch_size: int,
                 masking_ratio: float,
                 generator: torch.Generator,
                 reward_increase: bool):
        self.model = model
        self.num_samples = num_samples
        self.masking_ratio = masking_ratio
        self.generator = generator
        self.patch_size = patch_size
        self.reward_increase = reward_increase
        self.last_reward = 0

    def __call__(self, img: torch.Tensor, sampled_img: torch.Tensor) -> float:

        batch_reward = torch.zeros(self.num_samples, dtype=torch.float32)

        for i in range(self.num_samples):
            masked_sampled_img = self.sampled_img_random_masking(sampled_img)
            with torch.no_grad():
                outputs = self.model(img, masked_sampled_img)
            loss = outputs.loss
            reward_i = 1 / (1 + loss)
            batch_reward[i] = reward_i

        new_reward = batch_reward.sum().item()
        if self.reward_increase:
            new_reward = new_reward - self.last_reward
            self.last_reward = new_reward

        return new_reward

    def sampled_img_random_masking(self, sampled_img: torch.Tensor) -> torch.Tensor:

        B, C, H, W = sampled_img.shape
        x = torch.clone(sampled_img)
        x = x.permute(0, 2, 3, 1)

        kc, kh = self.patch_size, self.patch_size  # kernel size
        dc, dh = self.patch_size, self.patch_size  # stride

        patches = x.unfold(1, kc, dc).unfold(2, kh, dh)
        nan_mask = torch.isnan(patches)
        patch_nan_mask = nan_mask.any(dim=(3, 4, 5))
        valid_patches = ~patch_nan_mask
        valid_indices = torch.nonzero(valid_patches, as_tuple=True)

        num_valid = valid_indices[0].shape[0]  # Count of valid patches, error
        num_to_mask = int(self.masking_ratio * num_valid)  # Number of patches to mask

        mask_indices = torch.randperm(num_valid, generator=self.generator)[:num_to_mask]
        selected_patches = tuple(idx[mask_indices] for idx in valid_indices)  # Extract selected patch indices

        # Apply NaN masking
        patches[selected_patches] = float('nan')

        # Reassemble image from patches
        reconstructed = patches.permute(0, 3, 1, 4, 2, 5).view(B, C, H, W)

        return reconstructed


@dataclass
class ImageExplorationEnvConfig:
    device: str = DEVICE
    seed: int = None
    img_batch_size: int = 1
    sensor_size: int = 2 * 16
    steps_until_termination: int = 30
    interval_reward_assignment: int = 5
    v_max_x: int = sensor_size
    v_max_y: int = sensor_size
    v_max_z: int = 0

    # Set during execution
    img_sensor_ratio: float = None
    img: torch.Tensor = None
    reward_function: RewardFunction = None


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
            self.sensor_height = env_config.sensor_size
            self.sensor_width = env_config.sensor_size

        self._sensor_min_pos = torch.zeros((self.batch_size, 2), dtype=torch.int32)

        self.max_sensor_h = (self.img_height - self.sensor_height) // self.sensor_height + 1
        self.max_sensor_w = (self.img_width - self.sensor_width) // self.sensor_width + 1

        self.__sensor_pos = torch.stack([
            torch.randint(0, self.max_sensor_h, (self.batch_size,), dtype=torch.int32, generator=self.generator) * self.sensor_height,
            torch.randint(0, self.max_sensor_w, (self.batch_size,), dtype=torch.int32, generator=self.generator) * self.sensor_width,
        ], dim=1)

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
            low=-3.0,  # Bounds are a conservative estimate based on ImageNet images normalization params
            high=3.0,
            shape=(self.batch_size, 3, 224, 224),
            dtype=np.float32,
            seed=self.seed
        )
        self.action_space = spaces.Discrete(6)

    def _decode_action(self, action: int) -> np.ndarray:
        moves = np.array([
            [self.v_max_x, 0, 0], [-self.v_max_x, 0, 0],  # dx
            [0, self.v_max_y, 0], [0, -self.v_max_y, 0],  # dy
            [0, 0, self.v_max_z], [0, 0, -self.v_max_z]   # dz
        ])
        move = np.array([moves[action]])  # Add batch dim

        return move

    def _get_obs(self) -> np.ndarray:

        obs = torch.nan_to_num(self.sampled_img, nan=0.0).cpu().numpy()

        return obs

    def _get_info(self):  # TODO: Is there any info that I should be giving?
        info = {}

        return info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed != None:
            super().reset(seed=seed, options=options)

        self.__sensor_pos = torch.stack([
            torch.randint(0, self.max_sensor_h, (self.batch_size,), dtype=torch.int32, generator=self.generator) * self.sensor_height,
            torch.randint(0, self.max_sensor_w, (self.batch_size,), dtype=torch.int32, generator=self.generator) * self.sensor_width,
        ], dim=1)

        self.sampled_img = torch.full_like(self.img, float("nan"), device=self.device)
        self._sampled_kernel_size_mask = torch.full_like(
            self.img,
            fill_value=self.img_height // self.sensor_height,  # Max possible kernel size
            dtype=torch.int32
        )
        self._kernel_size = torch.ones(self.batch_size, dtype=torch.int32)
        self._reward_function.last_reward = 0
        self._step_count = 0

        self._update_sampled_img()  # Initial env observation is without movement
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray, eval: bool = False) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = self._decode_action(action)
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
