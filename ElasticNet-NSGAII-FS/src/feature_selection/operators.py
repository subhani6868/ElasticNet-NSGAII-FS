from __future__ import annotations
import numpy as np

def guided_init_population(N: int, d: int, w: np.ndarray, gamma: float, rng: np.random.Generator) -> np.ndarray:
    # Eq (9): p_j = gamma*w_j + (1-gamma)/d
    p = gamma * w + (1.0 - gamma) * (1.0 / float(d))
    P = rng.random((N, d)) < p[None, :]
    P = P.astype(np.int8)

    # Repair empty chromosomes
    best = int(np.argmax(w))
    empty = np.where(P.sum(axis=1) == 0)[0]
    if empty.size > 0:
        P[empty, best] = 1
    return P

def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    d = a.size
    if d <= 2:
        return a.copy(), b.copy()
    cx = int(rng.integers(1, d-1))
    c1 = np.concatenate([a[:cx], b[cx:]])
    c2 = np.concatenate([b[:cx], a[cx:]])
    return c1, c2

def guided_mutation(s: np.ndarray, w: np.ndarray, p_base: float, rng: np.random.Generator) -> np.ndarray:
    # Eq (10): p_mut_j = p_base * (1 - w_j)
    pm = p_base * (1.0 - w)
    flips = rng.random(s.size) < pm
    out = s.copy()
    out[flips] = 1 - out[flips]
    # Repair empty
    if out.sum() == 0:
        out[int(np.argmax(w))] = 1
    return out
