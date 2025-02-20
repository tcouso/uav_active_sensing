from pathlib import Path
import torch


def load_model_state_dict(weights_path: str, device: str):

    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model file not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location=torch.device(device), weights_only=True)

    print(f"State dict loaded from {weights_path} onto {device}")

    return state_dict
