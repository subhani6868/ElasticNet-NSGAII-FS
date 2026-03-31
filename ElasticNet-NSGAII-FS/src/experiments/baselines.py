from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.evaluation import evaluate_subset_cv
from src.feature_selection.elastic_net import elastic_net_priors
from src.feature_selection.nsga2_core import nsga2_guided
from src.feature_selection.objectives import objectives

@dataclass
class BaselineResult:
    name: str
    s: np.ndarray
    f1: float
    f2: float
    f3: float
    mean_accuracy: float
    chosen_k: Optional[int] = None

def _topk_subset(scores: np.ndarray, k: int) -> np.ndarray:
    d = scores.size
    k = int(min(max(k, 1), d))
    idx = np.argsort(-scores)[:k]
    s = np.zeros(d, dtype=np.int8)
    s[idx] = 1
    return s

def _choose_best_k(X: np.ndarray, y: np.ndarray, scores: np.ndarray, k_grid: List[int], cfg: dict, name: str) -> BaselineResult:
    best_acc = -1.0
    best_s = None
    best_k = None
    for k in k_grid:
        s = _topk_subset(scores, int(k))
        res = evaluate_subset_cv(X, y, s, cfg)
        if res.mean_acc > best_acc:
            best_acc = float(res.mean_acc)
            best_s = s
            best_k = int(min(max(k, 1), scores.size))
    f1, f2, f3 = objectives(X, y, best_s, cfg)
    return BaselineResult(
        name=name,
        s=best_s,
        f1=f1, f2=f2, f3=f3,
        mean_accuracy=float(1.0 - f1),
        chosen_k=best_k
    )

def baseline_elasticnet_topk(X: np.ndarray, y: np.ndarray, cfg: dict, k_grid: List[int]) -> BaselineResult:
    pri = elastic_net_priors(X, y, cfg)
    return _choose_best_k(X, y, pri.w, k_grid, cfg, "elasticnet_topk")

def baseline_mi_topk(X: np.ndarray, y: np.ndarray, cfg: dict, k_grid: List[int]) -> BaselineResult:
    Xs = StandardScaler().fit_transform(X)
    mi = mutual_info_classif(Xs, y, random_state=int(cfg["seed"]))
    return _choose_best_k(X, y, mi, k_grid, cfg, "mi_topk")

def baseline_random_topk(X: np.ndarray, y: np.ndarray, cfg: dict, k_grid: List[int]) -> BaselineResult:
    rng = np.random.default_rng(int(cfg["seed"]))
    scores = rng.random(X.shape[1])
    return _choose_best_k(X, y, scores, k_grid, cfg, "random_topk")

def baseline_lasso_logreg(X: np.ndarray, y: np.ndarray, cfg: dict) -> BaselineResult:
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=8000,
        n_jobs=None,
        random_state=int(cfg["seed"])
    )
    clf.fit(Xs, y)
    coef = np.abs(clf.coef_).ravel()
    s = (coef > 1e-12).astype(np.int8)
    if s.sum() == 0:
        s[int(np.argmax(coef))] = 1
    f1, f2, f3 = objectives(X, y, s, cfg)
    return BaselineResult(
        name="lasso_logreg",
        s=s,
        f1=f1, f2=f2, f3=f3,
        mean_accuracy=float(1.0 - f1),
        chosen_k=int(s.sum())
    )

def baseline_nsga2_vanilla(X: np.ndarray, y: np.ndarray, cfg: dict) -> BaselineResult:
    cfg2 = dict(cfg)
    cfg2["nsga2"] = dict(cfg["nsga2"])
    cfg2["nsga2"]["guidance_gamma"] = 0.0  # remove guidance
    d = X.shape[1]
    w_uniform = np.ones(d, dtype=float) / float(d)
    pareto = nsga2_guided(X, y, w_uniform, cfg2)

    F = np.array([ind.f for ind in pareto], dtype=float)
    mins = F.min(axis=0); maxs = F.max(axis=0)
    denom = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
    Fn = (F - mins) / denom
    chosen = int(np.argmin(Fn.sum(axis=1)))
    s = pareto[chosen].s
    f1, f2, f3 = objectives(X, y, s, cfg)
    return BaselineResult(
        name="nsga2_vanilla",
        s=s,
        f1=f1, f2=f2, f3=f3,
        mean_accuracy=float(1.0 - f1),
        chosen_k=int(s.sum())
    )

def run_baselines(X: np.ndarray, y: np.ndarray, cfg: dict) -> List[BaselineResult]:
    bcfg = cfg.get("baselines", {}) or {}
    methods = [str(m).lower() for m in bcfg.get("methods", [])]
    k_grid = [int(k) for k in bcfg.get("k_grid", [10, 20, 50, 100, 200])]
    out: List[BaselineResult] = []
    for m in methods:
        if m == "nsga2_vanilla":
            out.append(baseline_nsga2_vanilla(X, y, cfg))
        elif m == "elasticnet_topk":
            out.append(baseline_elasticnet_topk(X, y, cfg, k_grid))
        elif m == "lasso_logreg":
            out.append(baseline_lasso_logreg(X, y, cfg))
        elif m == "mi_topk":
            out.append(baseline_mi_topk(X, y, cfg, k_grid))
        elif m == "random_topk":
            out.append(baseline_random_topk(X, y, cfg, k_grid))
        else:
            raise ValueError(f"Unknown baseline method: {m}")
    return out

def baselines_to_frame(results: List[BaselineResult], d: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "method": r.name,
        "d": int(d),
        "n_selected": int(np.count_nonzero(r.s)),
        "chosen_k": (None if r.chosen_k is None else int(r.chosen_k)),
        "f1_error": float(r.f1),
        "f2_size_ratio": float(r.f2),
        "f3_instability": float(r.f3),
        "mean_accuracy": float(r.mean_accuracy),
    } for r in results])
