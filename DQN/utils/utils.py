import torch

def get_device() -> torch.device:
    # Check if the device can run cude or cpu and prio cuda.

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
