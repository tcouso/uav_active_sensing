import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from uav_active_sensing.config import DEVICE  # Assumes DEVICE is defined here, e.g. "cuda" or "cpu"

class CustomResNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        # Initialize the base feature extractor
        super(CustomResNetFeatureExtractor, self).__init__(observation_space, features_dim)

        # Load a pretrained ResNet18 model
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False  # Freeze the weights
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # Adaptive pooling to ensure fixed feature size
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Ensures (B, 512, 1, 1)

        # Update feature dimension
        self._features_dim = features_dim

    def forward(self, observations):
        """
        observations: expect a tensor of shape (batch_size, num_images, channels, height, width)
        """
        batch_size, num_images, C, H, W = observations.shape

        # Flatten the first two dimensions to process all images at once
        observations = observations.view(-1, C, H, W)  # Shape: (batch_size * num_images, C, H, W)

        # Extract features using ResNet
        features = self.resnet(observations)  # Shape: (batch_size * num_images, 512, H', W')
        features = self.global_avg_pool(features)  # Shape: (batch_size * num_images, 512, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten: (batch_size * num_images, 512)

        # Reshape back to (batch_size, num_images, feature_dim)
        features = features.view(batch_size, num_images, -1)  # Shape: (batch_size, num_images, 512)

        # Aggregate features (mean pooling across images)
        aggregated_features = features.mean(dim=1)  # Shape: (batch_size, 512)

        return aggregated_features

if __name__ == "__main__":
    # Create an instance of the feature extractor and move it to the specified DEVICE
    feature_extractor = CustomResNetFeatureExtractor(observation_space=None).to(DEVICE)
    
    # Verify the model is on the correct device
    print("Model is on device:", next(feature_extractor.parameters()).device)
    for name, param in feature_extractor.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
