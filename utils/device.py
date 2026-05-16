import torch

def get_device():
    """
    Automatically detect and return the best available device.
    Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def print_device_info():
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
