import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from uav_active_sensing.config import DEVICE


class CustomResNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomResNetFeatureExtractor, self).__init__(observation_space, features_dim)
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Freeze the weights
        for param in resnet.parameters():
            param.requires_grad = False  
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, observations):
        """
        observations: expect a tensor of shape (batch_size, num_images, channels, height, width)
        """
        batch_size, num_images, C, H, W = observations.shape
        observations = observations.view(-1, C, H, W)
        features = self.resnet(observations)
        features = features.view(batch_size, num_images, -1)
        aggregated_features = features.mean(dim=1)

        return aggregated_features


if __name__ == "__main__":
    feature_extractor = CustomResNetFeatureExtractor(observation_space=None).to(DEVICE)

    # Verify the model is on the correct device
    print("Model is on device:", next(feature_extractor.parameters()).device)
    for name, param in feature_extractor.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
