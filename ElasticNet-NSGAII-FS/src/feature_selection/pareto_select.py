from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from src.feature_selection.objectives import normalize_objectives

@dataclass
class ParetoDecision:
    s_star: np.ndarray
    utilities: np.ndarray
    knee_scores: np.ndarray
    chosen_index: int

def _knee_scores(Fn: np.ndarray) -> np.ndarray:
    # Fn: normalized objectives (n,3)
    # Use distance to line between extreme solutions on objective 1 and objective 2 (general heuristic)
    a = Fn[np.argmin(Fn[:, 0])]
    b = Fn[np.argmin(Fn[:, 1])]
    ab = b - a
    denom = np.linalg.norm(ab) if np.linalg.norm(ab) != 0 else 1.0
    # For each point p: distance to line through a,b in 3D using cross product magnitude / |ab|
    scores = []
    for p in Fn:
        ap = a - p
        # cross product for 3D vectors
        cp = np.cross(ab, ap)
        scores.append(np.linalg.norm(cp) / denom)
    return np.asarray(scores, dtype=float)

def select_final_subset(pareto_s: List[np.ndarray], pareto_F: np.ndarray, omega: Tuple[float,float,float], top_k: int) -> ParetoDecision:
    Fn = normalize_objectives(pareto_F)  # Eq (11)
    omega = np.asarray(omega, dtype=float)
    omega = omega / omega.sum()

    utilities = Fn @ omega  # Eq (12) (minimize)
    knee = _knee_scores(Fn) # Eq (13) heuristic

    # candidate set: top_k by knee (largest)
    k = int(min(max(top_k, 1), len(pareto_s)))
    cand_idx = np.argsort(-knee)[:k]
    chosen = cand_idx[np.argmin(utilities[cand_idx])]

    return ParetoDecision(
        s_star=pareto_s[int(chosen)],
        utilities=utilities,
        knee_scores=knee,
        chosen_index=int(chosen),
    )
