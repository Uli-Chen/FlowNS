"""
Compatibility patches for RecBole with modern PyTorch (>=2.6).

Import this module BEFORE importing recbole to apply patches.
"""

import functools
import torch


def patch_torch_load():
    """Patch torch.load to default weights_only=False for RecBole checkpoint loading.

    PyTorch 2.6 changed the default from False to True, which breaks RecBole's
    checkpoint loading (it stores Config objects via pickle).
    """
    _original_load = torch.load

    @functools.wraps(_original_load)
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load


def apply_all():
    """Apply all compatibility patches."""
    patch_torch_load()


# Auto-apply on import
apply_all()
