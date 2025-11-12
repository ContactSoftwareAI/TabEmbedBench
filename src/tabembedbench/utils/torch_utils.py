import logging

import torch


def get_device() -> torch.device:
    """
    Determines the appropriate device for computations based on GPU availability.

    This function checks if a CUDA-compatible GPU is available on the system. If
    a GPU is available, it returns a CUDA device. Otherwise, it returns the
    default CPU device.

    Returns:
        torch.device: The device object representing either a GPU (if available)
            or a CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def empty_gpu_cache(device: torch.device) -> None:
    """Clears the GPU cache for the specified device.

    This function frees up GPU memory by emptying the cache for the specified
    device. If the device is CUDA, it uses PyTorch's `torch.cuda.empty_cache`.
    If the device is Metal Performance Shaders (MPS), it uses
    `torch.mps.empty_cache`.

    Args:
        device (torch.device): The target GPU device for which the memory
            cache needs to be cleared.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def log_gpu_memory(logger: logging.Logger):
    message_lines = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3

            message_lines.append(f"GPU {i}:")
            message_lines.append(f"Allocated: {allocated:.2f} GB")
            message_lines.append(f"Reserved: {reserved:.2f} GB")
            message_lines.append(f"Max Allocated: {max_allocated:.2f} GB")
    else:
        message_lines.append(f"No GPU available")

    message = "\n".join(message_lines)

    logger.info(message)
