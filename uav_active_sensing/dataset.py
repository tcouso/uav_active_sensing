from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
import requests
import zipfile
from torchvision import datasets

from uav_active_sensing.config import TINY_IMAGENET_URL, EXTERNAL_DATA_DIR

app = typer.Typer()


def download_and_extract_tiny_imagenet(target_dir: Path):
    """Downloads and extracts the Tiny ImageNet dataset."""
    zip_path = target_dir / "tiny-imagenet-200.zip"

    if not zip_path.exists():
        logger.info("Downloading Tiny ImageNet dataset...")
        response = requests.get(TINY_IMAGENET_URL, stream=True)
        with open(zip_path, "wb") as f:
            total_size = int(response.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
        logger.info("Download completed.")

    extracted_dir = target_dir / "tiny-imagenet-200"
    if not extracted_dir.exists():
        logger.info("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        logger.info(f"Dataset extracted to {extracted_dir}")
    else:
        logger.info(f"Tiny ImageNet dataset already extracted at {extracted_dir}")


@app.command()
def main():

    # Download CIFAR-10
    datasets.CIFAR10(root=EXTERNAL_DATA_DIR, train=True, download=True)
    datasets.CIFAR10(root=EXTERNAL_DATA_DIR, train=False, download=True)
    logger.info(f"CIFAR-10 datasets have been downloaded and stored in {EXTERNAL_DATA_DIR}")

    # Download and extract Tiny ImageNet
    download_and_extract_tiny_imagenet(EXTERNAL_DATA_DIR)
    logger.info(f"Tiny ImageNet dataset has been downloaded and processed in {EXTERNAL_DATA_DIR}")


if __name__ == "__main__":
    app()
