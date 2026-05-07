"""Torch serialization helpers for trusted release artifacts."""

from __future__ import annotations

from typing import Any

import torch


def load_trusted_torch_artifact(path: str, **kwargs: Any) -> Any:
    """Load a trusted project artifact across PyTorch serialization versions.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    M-DESIGN releases trusted PyG ``Data`` objects and predictor modules, so
    those artifacts must be loaded with ``weights_only=False``.
    """
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
