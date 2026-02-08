import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Use the first GPU (NVIDIA 4060)
    else:
        device = torch.device("cpu")  # Fallback to CPU
    return device