import torch

from .runtime import get_window_extension

_window_extension = get_window_extension()
from .window_patch import enable_fused_window_partition


shift_window_partition = _window_extension.shift_window_partition

__all__ = [
    "enable_fused_window_partition",
    "shift_window_partition",
]
