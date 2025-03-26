import numpy as np
import torch
import gymnasium as gym

from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from gymnasium import spaces

from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining
from uav_active_sensing.tiny_imagenet_utils import tiny_imagenet_single_img_collate_fn


class RewardFunction:
    def __init__(self,
                 model: ActViTMAEForPreTraining,
                 num_samples: int,
                 patch_size: int,
                 masking_ratio: float,
                 generator: torch.Generator,
                 reward_increase: bool,
                 mask_sample: bool,
                 ):
        self.model = model
        self.num_samples = num_samples
        self.masking_ratio = masking_ratio
        self.generator = generator
        self.patch_size = patch_size
        self.reward_increase = reward_increase
        self.mask_sample = mask_sample
        self.last_reward = 0

    def __call__(self, img: torch.Tensor, sampled_img: torch.Tensor) -> float:

        rewards = torch.zeros(self.num_samples, dtype=torch.float32)

        for i in range(self.num_samples):
            if self.mask_sample:
                masked_img = self.sampled_img_random_masking(sampled_img)
            else:
                masked_img = sampled_img

            with torch.no_grad():
                outputs = self.model(
                    img.unsqueeze(0),
                    masked_img.unsqueeze(0)
                )
            loss = outputs.loss
            reward_i = 1 / (1 + loss)
            rewards[i] = reward_i

        new_reward = rewards.mean().item()
        if self.reward_increase:
            new_reward = new_reward - self.last_reward
            self.last_reward = new_reward

        return new_reward

    def sampled_img_random_masking(self, sampled_img: torch.Tensor) -> torch.Tensor:

        C, H, W = sampled_img.shape
        x = torch.clone(sampled_img)
        x = x.permute(1, 2, 0)

        kc, kh = self.patch_size, self.patch_size  # kernel size
        dc, dh = self.patch_size, self.patch_size  # stride

        patches = x.unfold(0, kc, dc).unfold(1, kh, dh)
        nan_mask = torch.isnan(patches)
        patch_nan_mask = nan_mask.any(dim=(2, 3, 4))
        valid_patches = ~patch_nan_mask
        valid_indices = torch.nonzero(valid_patches, as_tuple=True)

        num_valid = valid_indices[0].shape[0]  # Count of valid patches, error
        num_to_mask = int(self.masking_ratio * num_valid)  # Number of patches to mask

        mask_indices = torch.randperm(num_valid, generator=self.generator)[:num_to_mask]
        selected_patches = tuple(idx[mask_indices] for idx in valid_indices)  # Extract selected patch indices

        # Apply NaN masking
        patches[selected_patches] = float('nan')

        # Reassemble image from patches
        reconstructed = patches.permute(2, 0, 3, 1, 4).view(C, H, W)

        return reconstructed


@dataclass
class ImageExplorationEnvConfig:
    device: str = None
    seed: int = None
    sensor_size: int = None
    steps_until_termination: int = None
    interval_reward_assignment: int = None
    v_max_x: int = None
    v_max_y: int = None
    reward_function: RewardFunction = None


class ImageExplorationEnv(gym.Env):

    def __init__(self, dataset: Dataset, seed: int, env_config: ImageExplorationEnvConfig) -> None:
        super().__init__()
        self.device: ImageExplorationEnvConfig = env_config.device
        self.seed: int = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.dataset: Dataset = dataset
        self.dataloader: DataLoader = DataLoader(dataset,
                                                 collate_fn=tiny_imagenet_single_img_collate_fn,
                                                 generator=self.generator,
                                                 shuffle=True,
                                                 )
        self.iterator = iter(self.dataloader)
        self.img: torch.Tensor = next(iter(self.dataloader))
        self.img_h: int = self.img.shape[1]
        self.img_w: int = self.img.shape[2]
        self.sensor_h = env_config.sensor_size
        self.sensor_w = env_config.sensor_size
        self._sensor_min_pos = torch.tensor([0, 0], dtype=torch.int32)
        self.num_sensors_per_img_h = ((self.img_h - self.sensor_h) // self.sensor_h) + 1
        self.num_sensors_per_img_w = ((self.img_w - self.sensor_w) // self.sensor_w) + 1
        self.__sensor_pos = torch.stack([
            torch.randint(0, self.num_sensors_per_img_h, (1, ), dtype=torch.int32, generator=self.generator) * self.sensor_h,
            torch.randint(0, self.num_sensors_per_img_w, (1, ), dtype=torch.int32, generator=self.generator) * self.sensor_w,
        ], dim=1).flatten()
        self.sampled_img = torch.full_like(self.img, float("nan"), device=self.device)
        self.v_max_x = env_config.v_max_x
        self.v_max_y = env_config.v_max_y
        self.max_steps = env_config.steps_until_termination
        self._step_count = 0
        self.reward_function = env_config.reward_function
        self._interval_reward_assignment = env_config.interval_reward_assignment
        self.observation_space = spaces.Dict({
            'sampled_img': spaces.Box(
                low=-3.0,  # Bounds are a conservative estimate based on ImageNet images normalization params
                high=3.0,
                shape=self.img.shape,
                dtype=np.float32,
                seed=self.seed
            ),
            'pos': spaces.Box(low=np.array([0, 0]),
                              high=np.array([self.img_h - 1, self.img_w - 1]),
                              shape=(2,),
                              dtype=int)

        })
        self.action_space = spaces.Discrete(4)

        # Init env correctly with reset logic
        _, _ = self.reset()

    def _decode_action(self, action: int) -> np.ndarray:
        moves = np.array([
            [self.v_max_x, 0, 0], [-self.v_max_x, 0, 0],  # dx
            [0, self.v_max_y, 0], [0, -self.v_max_y, 0],  # dy
        ])
        move = np.array([moves[action]]).flatten()

        return move

    def _get_obs(self) -> Dict[np.ndarray, np.ndarray]:
        sampled_img = torch.nan_to_num(self.sampled_img, nan=0.0).cpu().numpy()
        pos = self.__sensor_pos.cpu().numpy()

        obs = {
            'sampled_img': sampled_img,
            'pos': pos
        }

        return obs

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[np.ndarray, np.ndarray], Dict]:
        if seed != None:
            super().reset(seed=seed, options=options)

        try:
            new_img = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            new_img = next(self.iterator)

        self.img = new_img

        self.__sensor_pos = torch.stack([
            torch.randint(0, self.num_sensors_per_img_h, (1, ), dtype=torch.int32, generator=self.generator) * self.sensor_h,
            torch.randint(0, self.num_sensors_per_img_w, (1, ), dtype=torch.int32, generator=self.generator) * self.sensor_w,
        ], dim=1).flatten()

        self.sampled_img = torch.full_like(self.img, float("nan"), device=self.device)
        self.reward_function.last_reward = 0
        self._step_count = 0

        self._update_sampled_img()  # Initial env observation is without movement
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[np.ndarray, np.ndarray], float, bool, bool, Dict]:
        action = self._decode_action(action)
        self.move(action)
        observation = self._get_obs()

        if (self._step_count % self._interval_reward_assignment == 0) and (self._step_count > 0):
            reward = self.reward_function(self.img, self.sampled_img)

        else:
            reward = float(0)

        self._step_count += 1
        terminated = self._step_count > self.max_steps

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
        """Returns fov positions in top left bottom right format stacked over batch """

        offset = torch.tensor([self.sensor_h, self.sensor_w])
        bottom_right = self._sensor_pos + offset
        fov_bbox = torch.cat((self._sensor_pos, bottom_right))

        return fov_bbox

    @property
    def _sensor_max_pos(self) -> torch.Tensor:
        """
        Computes the maximum valid position for the sensor based on the FoV.

        """
        fov_bbox = self.fov_bbox

        fov_height = fov_bbox[2] - fov_bbox[0]
        fov_width = fov_bbox[3] - fov_bbox[1]

        sensor_max_height_pos = self.img_h - fov_height
        sensor_max_width_pos = self.img_w - fov_width

        sensor_max_pos = torch.tensor([self.img_h - self.sensor_h, self.img_w - self.sensor_w], dtype=torch.int32)
        sensor_pos = torch.tensor([sensor_max_height_pos, sensor_max_width_pos])
        sensor_pos = torch.clamp(sensor_pos, min=self._sensor_min_pos, max=sensor_max_pos)

        return sensor_pos

    @property
    def _sensor_pos(self) -> torch.Tensor:

        return self.__sensor_pos

    @_sensor_pos.setter
    def _sensor_pos(self, new_position: torch.Tensor) -> None:
        """
        Sets the sensor position, ensuring it stays within valid bounds.
        """
        new_position = torch.clamp(new_position, min=self._sensor_min_pos, max=self._sensor_max_pos)
        self.__sensor_pos = new_position

    def _update_sampled_img(self) -> None:
        """
        Updates the sampled image based on the current sensor position.
        """
        fov_bbox = self.fov_bbox

        top = fov_bbox[0].item()
        left = fov_bbox[1].item()
        bottom = fov_bbox[2].item()
        right = fov_bbox[3].item()

        obs = self.img[:, top:bottom, left:right].clone()
        self.sampled_img[:, top:bottom, left:right] = obs

    def move(self, action: torch.Tensor) -> None:
        """
        Moves the sensor and updates the observation.

        Args:
            dx (int): The change in the x-direction.
            dy (int): The change in the y-direction.
        """

        self._sensor_pos += action[:2]
        self._update_sampled_img()
