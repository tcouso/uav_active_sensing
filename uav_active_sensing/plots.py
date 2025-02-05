from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from uav_active_sensing.config import FIGURES_DIR, PROCESSED_DATA_DIR

import matplotlib.pyplot as plt


def visualize_tensor(tensor, batch_idx=0):
    """
    Visualizes a PyTorch tensor as an image.
    If the tensor has a batch dimension, a specific batch index can be selected.

    Args:
        tensor (torch.Tensor): The tensor to display. Shape can be (B, C, H, W), (C, H, W), or (H, W).
        batch_idx (int, optional): The batch index to visualize. Defaults to 0.
    """
    # Ensure tensor is at least 2D
    if tensor.dim() < 2:
        raise ValueError("Tensor must be at least 2D.")

    # Handle batch dimension
    if tensor.dim() == 4:  # Batch dimension present (B, C, H, W)
        tensor = tensor[batch_idx]  # Select the specified batch index
    elif tensor.dim() == 3:  # No batch dimension (C, H, W)
        pass  # Use as is
    else:  # 2D tensor (H, W)
        pass  # Use as is

    # If tensor is 3D, check if it represents RGB or grayscale
    if tensor.dim() == 3 and tensor.shape[0] in {3, 1}:  # RGB or grayscale
        tensor = tensor.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)

    # Plot the tensor as an image
    plt.figure(figsize=(6, 6))
    plt.imshow(tensor.numpy())
    plt.title(f"Tensor (Batch {batch_idx})" if tensor.dim() == 4 else "Tensor")
    plt.axis('off')  # Hide axis
    plt.show()


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
