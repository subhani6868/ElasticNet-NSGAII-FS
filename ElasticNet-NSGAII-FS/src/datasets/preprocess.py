from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

from src.utils.io import ensure_dir
from src.datasets.nips2003_loader import load_nips2003_dataset

NIPS_KEYS = {"arcene", "dexter", "dorothea", "gisette", "madelon"}

def _find_file(dirp: Path, candidates: list[str]) -> Optional[Path]:
    for c in candidates:
        p = dirp / c
        if p.exists():
            return p
    return None

def _load_generic_csv(raw_dir: Path, dataset_key: str, cfg: dict) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    p_data = _find_file(raw_dir, ["data.csv", "dataset.csv"])
    p_X = _find_file(raw_dir, ["X.csv", "x.csv"])
    p_y = _find_file(raw_dir, ["y.csv", "Y.csv"])

    if p_data is not None:
        df = pd.read_csv(p_data)
        target_col = cfg["datasets"][dataset_key].get("target_col", "y") or "y"
        if target_col not in df.columns:
            raise ValueError(f"Expected label column '{target_col}' in {p_data}")
        y = df[target_col].to_numpy()
        X = df.drop(columns=[target_col]).to_numpy()
        feature_names = [c for c in df.columns if c != target_col]
        return X, y, feature_names

    if p_X is not None and p_y is not None:
        X = pd.read_csv(p_X, header=None).to_numpy()
        y = pd.read_csv(p_y, header=None).to_numpy().ravel()
        return X, y, None

    raise FileNotFoundError(f"Could not find data.csv or X.csv+y.csv under {raw_dir}")

def _load_svmlight(raw_dir: Path) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    p_svm = _find_file(raw_dir, ["data.svm", "data.svmlight", "train.svm", "train.svmlight", "data.txt"])
    if p_svm is None:
        raise FileNotFoundError(f"Could not find svmlight file under {raw_dir}")
    X, y = load_svmlight_file(str(p_svm))
    return X.toarray().astype(float), y.astype(int), None

def _load_raw(dataset_key: str, root: Path, cfg: dict) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    raw_dir = root / "data" / "raw" / dataset_key
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw data folder: {raw_dir}")

    if dataset_key.lower() in NIPS_KEYS:
        return load_nips2003_dataset(raw_dir, dataset_key)

    fmt = (cfg["datasets"][dataset_key].get("format", "csv") or "csv").lower()
    if fmt == "csv":
        return _load_generic_csv(raw_dir, dataset_key, cfg)
    if fmt == "svmlight":
        return _load_svmlight(raw_dir)
    raise ValueError(f"Unsupported dataset format: {fmt}")

def load_prepared_or_prepare(dataset_key: str, root: Path, cfg: dict) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    processed_dir = root / "data" / "processed" / dataset_key
    ensure_dir(processed_dir)

    pX = processed_dir / "X.npy"
    py = processed_dir / "y.npy"
    pfeat = processed_dir / "features.txt"

    if pX.exists() and py.exists():
        X = np.load(pX)
        y = np.load(py)
        feature_names = pfeat.read_text(encoding="utf-8").splitlines() if pfeat.exists() else None
        return X, y, feature_names

    X, y, feature_names = _load_raw(dataset_key, root, cfg)

    X = X.astype(float)
    var = X.var(axis=0)
    keep = var > 0
    X = X[:, keep]
    if feature_names is not None:
        feature_names = [fn for fn, k in zip(feature_names, keep) if k]

    np.save(pX, X)
    np.save(py, y)
    if feature_names is not None:
        pfeat.write_text("\n".join(feature_names), encoding="utf-8")

    return X, y, feature_names
