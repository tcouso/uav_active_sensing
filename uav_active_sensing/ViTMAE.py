import torch
from typing import Union, Optional, Tuple

from transformers.models.vit_mae.modeling_tf_vit_mae import (
    ViTMAEForPreTraining,
    ViTMAEForPreTrainingOutput,
)


class ActViTMAEForPreTraining(ViTMAEForPreTraining):

    def __init__(self, config, pixel_values: torch.Tensor):
        super.__init__(config)
        self.pixel_values = pixel_values
        self.forward = self.forward_loss_wrt_pixel_values
        self.vit.embeddings.random_masking = self.forward_with_two_input_tensors

    def forward_with_two_input_tensors(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        sampled_pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            sampled_pixel_values,  # Sampled pixel values are given to the encoder
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(
            latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding
        )
        logits = (
            decoder_outputs.logits
        )  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        # Loss is computed with respect to complete image
        loss = self.forward_loss(
            pixel_values, logits, mask, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def nan_masking(self, sequence, noise=None):
        """
        Mask patches where any value is NaN. Overriding of the original random masking function.

        Args:
            sequence (`torch.Tensor` of shape `(batch_size, sequence_length, dim)`):
                The patch embeddings.

        Returns:
            - `sequence_unmasked` (`torch.Tensor`): Patch embeddings without NaN patches.
            - `mask` (`torch.Tensor`): Binary mask where 1 indicates a masked patch.
            - `ids_restore` (`torch.Tensor`): Indices to restore original order.
        """
        _ = noise
        batch_size, seq_length, dim = sequence.shape

        # Identify patches with NaN values
        nan_mask = torch.isnan(sequence).any(
            dim=-1
        )  # Shape: (batch_size, seq_length), True if any NaN in patch

        # Keep only non-NaN patches
        ids_keep = (~nan_mask).nonzero(as_tuple=True)  # Get indices of patches to keep
        sequence_unmasked = sequence.clone()
        sequence_unmasked[nan_mask] = 0  # Replace NaN patches with zero

        # Create a binary mask: 0 for kept patches, 1 for masked patches
        mask = nan_mask.float()

        # Restore indices (identity since we're not shuffling)
        ids_restore = torch.arange(seq_length, device=sequence.device).repeat(batch_size, 1)

        return sequence_unmasked, mask, ids_restore
