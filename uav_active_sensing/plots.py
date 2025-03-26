import torch
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import ViTMAEForPreTraining

from uav_active_sensing.config import IMAGENET_MEAN, IMAGENET_STD
from uav_active_sensing.modeling.mae.act_vit_mae import ActViTMAEForPreTraining

# From: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb
def show_image(image, title=""):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis("off")
    return


def visualize_mae_reconstruction(pixel_values: torch.Tensor, model: ViTMAEForPreTraining, save_path: Path = None, show: bool = True):
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', pixel_values).detach().cpu()

    # Masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # Make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], f"reconstruction (MSE: {outputs.loss:.6f})")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()  # Close to free memory when not displaying


def visualize_act_mae_reconstruction(pixel_values: torch.Tensor,
                                     sampled_pixel_values: torch.Tensor,
                                     masked_sampled_pixel_values: torch.Tensor,
                                     model: ActViTMAEForPreTraining,
                                     save_path: Path = None,
                                     show: bool = True
                                     ):

    # Forward pass
    outputs = model(pixel_values, masked_sampled_pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    # Visualize the mask
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

    x = torch.einsum("nchw->nhwc", pixel_values).detach().cpu()

    # Sampled image
    im_sampled = torch.nan_to_num(sampled_pixel_values.detach().cpu().permute(0, 2, 3, 1))

    # Masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # Make the plt figure larger
    plt.rcParams["figure.figsize"] = [24, 24]

    plt.subplot(1, 5, 1)
    show_image(x[0], "original")

    plt.subplot(1, 5, 2)
    show_image(im_sampled[0], "sampled")

    plt.subplot(1, 5, 3)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 5, 4)
    show_image(y[0], f"reconstruction (MSE: {outputs.loss:.6f})")

    plt.subplot(1, 5, 5)
    show_image(im_paste[0], "reconstruction + visible")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_tensor(tensor, batch_idx=0, save_path: Path = None, show: bool = True):
    """
    Visualizes a PyTorch tensor as an image.
    If the tensor has a batch dimension, a specific batch index can be selected.

    Args:
        tensor (torch.Tensor): The tensor to display. Shape can be (B, C, H, W), (C, H, W), or (H, W).
        batch_idx (int, optional): The batch index to visualize. Defaults to 0.
    """
    if tensor.dim() < 2:
        raise ValueError("Tensor must be at least 2D.")

    if tensor.dim() == 4:  # Batch dimension present (B, C, H, W)
        tensor = tensor[batch_idx]  # Select the specified batch index
    elif tensor.dim() == 3:  # No batch dimension (C, H, W)
        pass  # Use as is
    else:  # 2D tensor (H, W)
        pass  # Use as is

    if tensor.dim() == 3 and tensor.shape[0] in {3, 1}:  # RGB or grayscale
        tensor = tensor.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)

    plt.figure(figsize=(6, 6))
    plt.imshow(tensor.cpu().numpy())
    plt.title(f"Tensor (Batch {batch_idx})" if tensor.dim() == 4 else "Tensor")
    plt.axis("off")  # Hide axis

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
