import torch
import torch.nn.functional as F
from typing import Tuple


class ImageEnv:
    def __init__(
        self, image: torch.Tensor,
        img_FoV_ratio: int,
        min_altitude: int,
        max_altitude: int,
        device: str = "cpu"
    ) -> None:
        """
        Initializes the ImageEnv class with the given image and sensor parameters.

        Args:
            image (torch.Tensor): The input image tensor with shape (C, H, W), where
                                  C is the number of channels, H is the height, and
                                  W is the width.
            img_FoV_ratio (int): The ratio that defines the sensor's Field of View (FoV).
            min_altitude (int): The minimum altitude of the sensor.
            max_altitude (int): The maximum altitude of the sensor.
            device (str, optional): The device to run the computation on ('cpu' or 'cuda').
                                    Defaults to 'cpu'.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.img = image.to(self.device)
        self.img_height, self.img_width = image.shape[1:]

        self.img_sensor_ratio = img_FoV_ratio
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude

        self.sensor_height = self.img_height // self.img_sensor_ratio
        self.sensor_width = self.img_width // self.img_sensor_ratio

        self._sensor_pos = [0, 0, self.min_altitude]
        self.sensor_min_pos = [0, 0, self.min_altitude]

        self.max_zoom_level = 1
        self.min_zoom_level = max(
            self.img_height // self.sensor_height,
            self.img_width // self.sensor_width)

        self.sampled_img = torch.full_like(self.img, float('nan'), device=self.device)
        self.sample_img_altitude_mask = torch.full_like(self.img, fill_value=self.max_altitude, dtype=torch.int32)

    @property
    def fov_bbox(self) -> Tuple[int, int, int, int]:
        """
        Calculates the bounding box of the sensor's Field of View (FoV) based on the sensor position and zoom level.

        Returns:
            Tuple[int, int, int, int]: A tuple (top, bottom, left, right) representing
                                        the bounding box of the FoV.
        """
        zoomed_size_height = int(self.sensor_height * self.zoom_level)
        zoomed_size_width = int(self.sensor_width * self.zoom_level)

        top = max(0, self.sensor_pos[0])
        left = max(0, self.sensor_pos[1])
        bottom = min(self.img_height, self.sensor_pos[0] + zoomed_size_height)
        right = min(self.img_width, self.sensor_pos[1] + zoomed_size_width)

        return [top, bottom, left, right]

    @property
    def sensor_max_pos(self) -> Tuple[int, int, int]:
        """
        Calculates the maximum possible position of the sensor, ensuring the sensor stays within bounds.

        Returns:
            Tuple[int, int, int]: A tuple (max_height, max_width, max_altitude) representing
                                   the maximum sensor position along each axis.
        """
        top, bottom, left, right = self.fov_bbox

        fov_height = bottom - top
        fov_width = right - left
        sensor_max_height = self.img_height - fov_height
        sensor_max_width = self.img_width - fov_width

        return (sensor_max_height, sensor_max_width, self.max_altitude)

    @property
    def sensor_pos(self) -> Tuple[int, int, int]:
        """
        Gets the current position of the sensor.

        Returns:
            Tuple[int, int, int]: A tuple (x, y, z) representing the sensor's position in space.
        """
        return tuple(self._sensor_pos)

    @sensor_pos.setter
    def sensor_pos(self, new_position: Tuple[int, int, int]) -> None:
        """
        Sets the position of the sensor, ensuring that the new position is within the allowed bounds.

        Args:
            new_position (Tuple[int, int, int]): A tuple (x, y, z) representing the new position of the sensor.
        """
        x, y, z = new_position

        # z update first, in order to constrain x and y
        self._sensor_pos[2] = max(min(z, self.sensor_max_pos[2]), self.sensor_min_pos[2])

        # x and y update based on the new max positions
        self._sensor_pos[0] = max(min(x, self.sensor_max_pos[0]), self.sensor_min_pos[0])
        self._sensor_pos[1] = max(min(y, self.sensor_max_pos[1]), self.sensor_min_pos[1])

    @property
    def zoom_level(self) -> float:
        """
        Calculates the current zoom level based on the sensor's altitude.

        Returns:
            float: The zoom level of the sensor based on its altitude.
        """
        m = (self.min_zoom_level - self.max_zoom_level) / (self.max_altitude - self.min_altitude)
        b = self.max_zoom_level - m * self.min_altitude

        return m * self._sensor_pos[2] + b

    def _apply_blur(self, window: torch.Tensor) -> torch.Tensor:
        """
        Applies an averaging blur to the input window tensor while considering margin artifacts.

        Args:
            window (torch.Tensor): A tensor of shape (C, H, W), where C is the number of channels,
                                   H is the height, and W is the width of the image.

        Returns:
            torch.Tensor: A blurred tensor with the same shape as the input window.
        """
        kernel_size = int(self.zoom_level)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        padding = kernel_size // 2
        window_padded = F.pad(window, (padding, padding, padding, padding), mode='reflect')  # Apply reflection padding to avoid margin artifacts
        blurred = F.avg_pool2d(window_padded.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=0).squeeze(0)

        return blurred

    def _apply_higher_zoom_filter(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Updates the observed image based on the current sensor position and altitude.

        Args:
            obs (torch.Tensor): The current observation tensor of shape (C, H, W).

        Returns:
            torch.Tensor: The updated observation tensor.
        """
        top, bottom, left, right = self.fov_bbox

        previous_altitude_mask = self.sample_img_altitude_mask[:, top:bottom, left:right]
        current_altitude_mask = torch.full_like(previous_altitude_mask, fill_value=self.sensor_pos[2])

        # Altitude mask updating
        altitudes_to_update = current_altitude_mask < previous_altitude_mask
        self.sample_img_altitude_mask[:, top:bottom, left:right][altitudes_to_update] = current_altitude_mask[altitudes_to_update]

        # Observation updating
        prev_obs = self.sampled_img[:, top:bottom, left:right]
        obs_to_update = current_altitude_mask > previous_altitude_mask
        obs[obs_to_update] = prev_obs[obs_to_update]

        return obs

    def _observe(self) -> None:
        """
        Captures an observation from the image based on the current sensor position and zoom level.

        This method applies a blur if the zoom level is above the maximum zoom level and updates
        the sampled image with the new observation.
        """
        top, bottom, left, right = self.fov_bbox
        obs = self.img[:, top:bottom, left:right].clone()

        if self.zoom_level > self.max_zoom_level:
            obs = self._apply_blur(obs)
            obs = obs.squeeze(0)

        obs = self._apply_higher_zoom_filter(obs)
        self.sampled_img[:, top:bottom, left:right] = obs

    def move(self, dx: int, dy: int, dz: int) -> None:
        """
        Moves the sensor by the specified amount along each axis and updates the observation.

        Args:
            dx (int): The amount to move along the x-axis.
            dy (int): The amount to move along the y-axis.
            dz (int): The amount to move along the z-axis (altitude).
        """
        self.sensor_pos = (
            self.sensor_pos[0] + dy,
            self.sensor_pos[1] + dx,
            self.sensor_pos[2] + dz
        )
        self._observe()
