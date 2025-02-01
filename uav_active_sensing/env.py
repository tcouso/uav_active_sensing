import random as rd
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional

class ImageEnv:
    """
    A class to simulate a sensor moving over an image, with adjustable position and blurring.

    Attributes:
        device (str): The device (e.g., 'cpu' or 'cuda') where the image and computations reside.
        img (torch.Tensor): The input image tensor.
        img_height (int): The height of the image.
        img_width (int): The width of the image.
        img_sensor_ratio (int): The ratio of the image size to the sensor size.
        sensor_height (int): The height of the sensor.
        sensor_width (int): The width of the sensor.
        sampled_img (torch.Tensor): The image tensor with sampled regions.
    """

    def __init__(
        self, 
        image: torch.Tensor, 
        img_FoV_ratio: int, 
        device: Optional[str] = "cpu"
    ) -> None:
        """
        Initializes the ImageEnv with an image and sensor parameters.

        Args:
            image (torch.Tensor): The input image tensor of shape (C, H, W).
            img_FoV_ratio (int): The ratio of the image size to the sensor size.
            device (Optional[str]): The device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.img = image.to(self.device)
        self.img_height, self.img_width = image.shape[1:]

        self.img_sensor_ratio = img_FoV_ratio

        self.sensor_height = self.img_height // self.img_sensor_ratio
        self.sensor_width = self.img_width // self.img_sensor_ratio

        self._sensor_min_pos = [0, 0]
        self.__sensor_pos = [rd.randint(0, self.img_height,), rd.randint(0, self.img_width,)]

        self._min_kernel_size = 1
        self._max_kernel_size = self.img_height - self.sensor_height + 1
        self._max_kernel_size = (
            self._max_kernel_size if self._max_kernel_size % 2 == 1 
            else self._max_kernel_size - 1
        )
        self.__kernel_size = self._min_kernel_size

        self._sampled_kernel_size_mask = torch.full_like(
            self.img, fill_value=self._max_kernel_size, dtype=torch.int32
        )
        self.sampled_img = torch.full_like(self.img, float('nan'), device=self.device)

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
    def sensor_max_pos(self) -> Tuple[int, int]:
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
        self.__sensor_pos[0] = max(min(x, self.sensor_max_pos[0]), self._sensor_min_pos[0])
        self.__sensor_pos[1] = max(min(y, self.sensor_max_pos[1]), self._sensor_min_pos[1])

    @property
    def _kernel_size(self) -> int:
        """
        Gets the current kernel size.

        Returns:
            int: The current kernel size.
        """
        return self.__kernel_size

    @_kernel_size.setter
    def _kernel_size(self, kernel_size: int) -> None:
        """
        Sets the kernel size, ensuring it stays within valid bounds.

        Args:
            kernel_size (int): The new kernel size.
        """
        max_kernel_size_from_sensor_pos = min(self.img_height, self.img_width) - max(self._sensor_pos)
        max_kernel_size_from_sensor_pos = (
            max_kernel_size_from_sensor_pos if max_kernel_size_from_sensor_pos % 2 == 1 
            else max_kernel_size_from_sensor_pos - 1
        )
        self.__kernel_size = max(min(kernel_size, max_kernel_size_from_sensor_pos), self._min_kernel_size)

    def _apply_blur(self, window: torch.Tensor) -> torch.Tensor:
        """
        Applies an averaging blur to the input window tensor while considering margin artifacts.

        Args:
            window (torch.Tensor): A tensor of shape (C, H, W), where C is the number of channels,
                                H is the height, and W is the width of the image.

        Returns:
            torch.Tensor: A blurred tensor with the same shape as the input window.
        """
        padding = self._kernel_size // 2
        window_padded = F.pad(window, (padding, padding, padding, padding), mode='reflect')
        blurred = F.avg_pool2d(window_padded.unsqueeze(0), kernel_size=self._kernel_size, stride=1, padding=0).squeeze(0)

        assert blurred.shape == window.shape

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

        prev_mask = self._sampled_kernel_size_mask[:, top:bottom, left:right]
        curr_mask = torch.full_like(prev_mask, fill_value=self._kernel_size)

        # Blur mask update
        updated_mask = curr_mask < prev_mask
        self._sampled_kernel_size_mask[:, top:bottom, left:right][updated_mask] = curr_mask[updated_mask]

        # Observation update
        prev_obs = self.sampled_img[:, top:bottom, left:right]
        obs_to_update = curr_mask > prev_mask
        obs[obs_to_update] = prev_obs[obs_to_update]

        return obs

    def _observe(self) -> None:
        """
        Updates the sampled image based on the current sensor position and kernel size.
        """
        top, bottom, left, right = self.fov_bbox
        obs = self.img[:, top:bottom, left:right].clone()

        if self._kernel_size > self._min_kernel_size:
            obs = self._apply_blur(obs)
            obs = obs.squeeze(0)

        obs = self._filter_high_blur_obs(obs)

        self.sampled_img[:, top:bottom, left:right] = obs

    def move(self, dx: int, dy: int, dz: int) -> None:
        """
        Moves the sensor and updates the observation.

        Args:
            dx (int): The change in the x-direction.
            dy (int): The change in the y-direction.
            dz (int): The change in the kernel size.
        """
        self._sensor_pos = (
            self._sensor_pos[0] + dy,
            self._sensor_pos[1] + dx,
        )
        self._kernel_size += 2 * dz

        self._observe()