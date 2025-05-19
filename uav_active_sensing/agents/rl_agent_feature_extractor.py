from typing import Dict
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CustomResNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        resnet_features_dim: int = 512,
        pos_features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim=resnet_features_dim + pos_features_dim)

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        for param in resnet.parameters():
            param.requires_grad = False  # Freeze the weights

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        extractors = {}

        for key, subspace in observation_space.spaces.items():

            if key == "sampled_img":
                extractors[key] = self.resnet

            elif key == "pos":
                extractors[key] = nn.Linear(subspace.shape[0], pos_features_dim)

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: Dict[np.ndarray, np.ndarray]) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            feature = extractor(observations[key])

            # If the feature has more than 2 dimensions, flatten it (except the batch dimension)
            if feature.dim() > 2:
                feature = torch.flatten(feature, start_dim=1)

            encoded_tensor_list.append(feature)

        # Concatenate along the feature dimension (dim=1)
        joint_features = torch.cat(encoded_tensor_list, dim=1)

        return joint_features
