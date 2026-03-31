from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from src.datasets.preprocess import load_prepared_or_prepare

def load_dataset(dataset_key: str, root: Path, cfg: dict) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    # Returns X, y, feature_names
    return load_prepared_or_prepare(dataset_key, root, cfg)
