from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
import requests
import zipfile
from torchvision import datasets, transforms

from uav_active_sensing.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_PROCESSED_DIR = PROCESSED_DATA_DIR / "tiny_imagenet"


def download_and_extract_tiny_imagenet(target_dir: Path):
    """Downloads and extracts the Tiny ImageNet dataset."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "tiny-imagenet-200.zip"
    
    if not zip_path.exists():
        logger.info("Downloading Tiny ImageNet dataset...")
        response = requests.get(TINY_IMAGENET_URL, stream=True)
        with open(zip_path, "wb") as f:
            total_size = int(response.headers.get('content-length', 0))
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
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Download and process CIFAR-10
    cifar10_processed_dir = PROCESSED_DATA_DIR / "cifar10"
    datasets.CIFAR10(
        root=cifar10_processed_dir, train=True, download=True, transform=transform
    )
    datasets.CIFAR10(
        root=cifar10_processed_dir, train=False, download=True, transform=transform
    )
    logger.info(f"CIFAR-10 datasets have been downloaded and stored in {cifar10_processed_dir}")
    
    # Download and extract Tiny ImageNet
    download_and_extract_tiny_imagenet(TINY_IMAGENET_PROCESSED_DIR)
    logger.info(f"Tiny ImageNet dataset has been downloaded and processed in {TINY_IMAGENET_PROCESSED_DIR}")


if __name__ == "__main__":
    app()
