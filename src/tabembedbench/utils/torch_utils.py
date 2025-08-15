import torch

def get_device() -> torch.device:
    """
    Determines the appropriate PyTorch device to be used.

    This function checks the system's hardware capabilities and returns a PyTorch
    device object to represent the computing device (e.g., GPU or CPU) to be used
    for tensor operations. It prioritizes the devices in the following order:
    1. CUDA-enabled GPU (NVIDIA GPU).
    2. MPS-capable GPU (Apple Silicon GPU).
    3. CPU as the fallback.

    Returns:
        torch.device: The torch.device object representing the device that can be
        used for tensor computation.
    """
    if torch.is_cuda_available():
        return torch.device("cuda")
    elif torch.is_mps_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")