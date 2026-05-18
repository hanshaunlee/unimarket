"""Deterministic seeding across libraries."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and (if available) PyTorch.

    Sets PYTHONHASHSEED for child processes and toggles deterministic algorithms
    in PyTorch when requested.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is a hard dep, but stay defensive
        pass


def seeded_rng(seed: int) -> np.random.Generator:
    """Return a NumPy Generator with a fixed seed."""
    return np.random.default_rng(seed)
