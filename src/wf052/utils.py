"""Utility functions for reproducible random seeding and runtime logging."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across python, numpy, and pytorch for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_environment_versions() -> dict[str, str]:
    """Return dictionary of core dependency versions."""
    import geopandas
    import pyarrow
    import pytest
    import rasterio
    import torchvision
    import yaml
    import zarr

    return {
        "python": "3.11.15",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
        "geopandas": geopandas.__version__,
        "pyarrow": pyarrow.__version__,
        "zarr": zarr.__version__,
        "pyyaml": yaml.__version__,
        "pytest": pytest.__version__,
    }
