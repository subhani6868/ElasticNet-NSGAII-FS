from __future__ import annotations
import numpy as np
from src.models.evaluation import evaluate_subset_cv

def objectives(X: np.ndarray, y: np.ndarray, s: np.ndarray, cfg: dict) -> tuple[float, float, float]:
    # f1: 1 - mean accuracy
    # f2: |s|/d
    # f3: std accuracy across CV folds (instability proxy)
    res = evaluate_subset_cv(X, y, s, cfg)
    f1 = 1.0 - res.mean_acc
    f2 = float(np.count_nonzero(s)) / float(s.size)
    f3 = float(res.std_acc)
    return f1, f2, f3

def normalize_objectives(F: np.ndarray) -> np.ndarray:
    # F: (n, m)
    F = np.asarray(F, dtype=float)
    mins = F.min(axis=0)
    maxs = F.max(axis=0)
    denom = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
    return (F - mins) / denom
