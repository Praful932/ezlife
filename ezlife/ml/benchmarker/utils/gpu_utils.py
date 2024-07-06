"""gpu go brrr
later use pynvml, use runpod env vars till then
"""
import os
import warnings

def get_gpu_count():
    gpu_count = os.getenv("RUNPOD_GPU_COUNT")
    if gpu_count is None:
        warnings.warn("GPU count not found in env var RUNPOD_GPU_COUNT")
    return gpu_count

def get_gpu_name():
    gpu_name = os.getenv("RUNPOD_GPU_NAME")
    if gpu_name is None:
        warnings.warn("GPU name not found in env var RUNPOD_GPU_NAME")
    return gpu_name

