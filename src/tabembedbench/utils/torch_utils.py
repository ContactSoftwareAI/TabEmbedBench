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
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def empty_gpu_cache(device: torch.device) -> None:
    """
    Clears the GPU cache for the specified device.

    This function frees up GPU memory by emptying the cache for the specified
    device. If the device is CUDA, it uses PyTorch's `torch.cuda.empty_cache`.
    If the device is Metal Performance Shaders (MPS), it uses
    `torch.mps.empty_cache`.

    Args:
        device (torch.device): The target GPU device for which the memory
            cache needs to be cleared.

    Returns:
        None
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()