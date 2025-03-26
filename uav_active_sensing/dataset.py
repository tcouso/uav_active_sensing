import typer
from loguru import logger
from torchvision import datasets

from uav_active_sensing.config import EXTERNAL_DATA_DIR
from uav_active_sensing.tiny_imagenet_utils import download_and_extract_tiny_imagenet
app = typer.Typer()


@app.command()
def main():

    # CIFAR-10
    datasets.CIFAR10(root=EXTERNAL_DATA_DIR, train=True, download=True)
    datasets.CIFAR10(root=EXTERNAL_DATA_DIR, train=False, download=True)
    logger.info(f"CIFAR-10 datasets have been downloaded and stored in {EXTERNAL_DATA_DIR}")

    # Tiny ImageNet
    download_and_extract_tiny_imagenet(EXTERNAL_DATA_DIR)
    logger.info(f"Tiny ImageNet dataset has been downloaded and processed in {EXTERNAL_DATA_DIR}")


if __name__ == "__main__":
    app()
