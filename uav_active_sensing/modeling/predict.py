from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from uav_active_sensing.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


# TODO: Take trained agent, obtain mask of new images according to trained (fixed) policy. Store masks for visualization
# TODO: Take masked and actual image and generate predicted image with ViTMAE


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
