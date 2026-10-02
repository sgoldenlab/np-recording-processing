"""GPU memory helpers for long Kilosort4 batches (torch imported lazily).

Python-side cleanup only frees memory; it CANNOT recover a CUDA context corrupted by a kernel
that hit the display-watchdog timeout. See README ("CUDA watchdog") for the real fix.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import sleep


def clear_gpu_memory(sleep_time: float = 0, force_synchronize: bool = False) -> None:
    """Empty the CUDA cache. ``force_synchronize`` can hang if a kernel is stuck."""
    import torch

    if force_synchronize and torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if sleep_time > 0:
        sleep(sleep_time)


def reset_gpu_state(aggressive: bool = False, safe_mode: bool = True) -> None:
    """Between-session cleanup. ``safe_mode=True`` skips synchronize (avoids hanging)."""
    import gc

    import torch

    if not torch.cuda.is_available():
        return
    if not safe_mode:
        try:
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"(!) GPU synchronize failed: {e}")
    torch.cuda.empty_cache()
    if aggressive:
        gc.collect()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


@contextmanager
def gpu_memory_context(sleep_time: float = 10, force_sync: bool = False):
    """Clear GPU memory on entry (then sleep) and on exit."""
    clear_gpu_memory(sleep_time, force_synchronize=force_sync)
    try:
        yield
    finally:
        clear_gpu_memory(0, force_synchronize=force_sync)
