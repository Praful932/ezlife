import gc
import ctypes
import torch

def gc_cuda():
    """Gargage collect RAM & Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        torch.cuda.empty_cache()

def get_device():
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    return device