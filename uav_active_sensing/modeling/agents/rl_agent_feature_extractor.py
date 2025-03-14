from typing import Dict
import numpy as np

import torch
from torch import nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from uav_active_sensing.config import DEVICE
import gymnasium as gym

# TODO: Use this format in refactoring


# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super().__init__(observation_space, features_dim=1)

#         extractors = {}

#         total_concat_size = 0
#         # We need to know size of the output of this extractor,
#         # so go over all the spaces and compute output feature sizes
#         for key, subspace in observation_space.spaces.items():
#             if key == "image":
#                 # We will just downsample one channel of the image by 4x4 and flatten.
#                 # Assume the image is single-channel (subspace.shape[0] == 0)
#                 extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
#                 total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
#             elif key == "vector":
#                 # Run through a simple MLP
#                 extractors[key] = nn.Linear(subspace.shape[0], 16)
#                 total_concat_size += 16

#         self.extractors = nn.ModuleDict(extractors)

#         # Update the features dim manually
#         self._features_dim = total_concat_size

#     def forward(self, observations) -> torch.Tensor:
#         encoded_tensor_list = []

#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         return torch.cat(encoded_tensor_list, dim=1)


class CustomResNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, resnet_features_dim: int = 512, pos_features_dim: int = 64):
        super().__init__(observation_space, features_dim=resnet_features_dim)
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        for param in resnet.parameters():
            param.requires_grad = False  # Freeze the weights
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        extractors = {}

        for key, subspace in observation_space.spaces.items():

            if key == 'sampled_img':  # TODO: refactor this info a regular image observation

                def img_extractor(img_observations: np.ndarray):
                    batch_size, num_images, C, H, W = img_observations.shape
                    img_observations = img_observations.view(-1, C, H, W)
                    features = self.resnet(img_observations)
                    features = features.view(batch_size, num_images, -1)

                    return features

                extractors[key] = img_extractor

            elif key == 'pos':
                extractors[key] = nn.Linear(subspace.shape[0], pos_features_dim)

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: Dict[np.ndarray, np.ndarray]) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return torch.cat(encoded_tensor_list, dim=1)


if __name__ == "__main__":
    feature_extractor = CustomResNetFeatureExtractor(observation_space=None).to(DEVICE)

    # Verify the model is on the correct device
    print("Model is on device:", next(feature_extractor.parameters()).device)
    for name, param in feature_extractor.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
