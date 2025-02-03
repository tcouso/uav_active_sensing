import torch
from abc import ABC, abstractmethod

from uav_active_sensing.env import ImageEnv
from transformers.models.vit_mae.modeling_tf_vit_mae import ViTMAEEmbeddings


class ActiveMaskingStrategy(ViTMAEEmbeddings):

    def __init__(self, env: ImageEnv):
        """
        Initialize the ActiveMaskingStrategy with an optional environment variable.
        """
        self.env: ImageEnv = env
        self.pixel_values: torch.Tensor = None
        self.interpolate_pos_encoding: bool = None

    def active_masking(self, sequence: torch.Tensor, noise=None):
        # TODO: this should work as if it was a random sampling that ended having the same result that the policy
        # This is necessary in order to properly reconstruct the masked image

        batch_size, num_channels, height, width = self.pixel_values.shape

        embeddings = self.patch_embeddings(self.pixel_values, interpolate_pos_encoding=self.interpolate_pos_encoding)

        if self.interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = self.position_embeddings

        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.nan_masking(embeddings, noise)

        _ = noise  # Not used in this strategy
        batch_size, seq_length, dim = sequence.shape

        # Identify patches with NaN values
        nan_mask = torch.isnan(sequence).any(dim=-1)

        # Mask out NaN patches
        sequence_unmasked = sequence.clone()
        sequence_unmasked[nan_mask] = 0  # Replace NaN patches with zero

        # Create binary mask (1 = masked, 0 = kept)
        mask = nan_mask.float()

        ids_restore = torch.arange(seq_length, device=sequence.device).repeat(batch_size, 1)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, noise=None, interpolate_pos_encoding: bool = False):
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.pixel_values = pixel_values
        self.sampled_pixel_values = None  # Insert the env sampling logic here

        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = self.position_embeddings

        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.active_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore
