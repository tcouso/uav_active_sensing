from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import torch
import random
import numpy as np

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_DIR = EXTERNAL_DATA_DIR /  "tiny_imagenet/tiny-imagenet-200"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

CIFAR10_DIR = EXTERNAL_DATA_DIR / "cifar10/cifar-10-batches-py"

# Random seed setter function
SEED = 84
# def set_seed(seed: int = SEED):
#     """Set the random seed for reproducibility across multiple libraries."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# Select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Batch size
IMG_BATCH_SIZE = 1

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
