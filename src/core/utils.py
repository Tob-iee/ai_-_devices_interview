import os
import torch


_DEVICE_OVERRIDE = os.getenv("LLM_DEVICE", "").lower().strip()
_DTYPE_OVERRIDE  = os.getenv("LLM_DTYPE", "").lower().strip()

def _choose_device() -> str:
    if _DEVICE_OVERRIDE in {"cuda", "mps", "cpu"}:
        return _DEVICE_OVERRIDE
    if torch.cuda.is_available():
        return "cuda"          # single-GPU; for multi-GPU you can consider "auto"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _choose_dtype(device: str):
    if _DTYPE_OVERRIDE in {"float16", "fp16"}:
        return torch.float16
    if _DTYPE_OVERRIDE in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if _DTYPE_OVERRIDE in {"float32", "fp32"}:
        return torch.float32

    # sensible defaults per device
    if device == "cuda":
        return torch.float16   # fast & common on NVIDIA
    if device == "mps":
        # MPS handles fp16; bf16 support depends on macOS/PyTorch version.
        return torch.float16
    return torch.float32 