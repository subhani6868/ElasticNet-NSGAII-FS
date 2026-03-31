from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

@dataclass
class ENPriors:
    w: np.ndarray          # shape (d,)
    ranking: np.ndarray    # descending indices

def elastic_net_priors(X: np.ndarray, y: np.ndarray, cfg: dict) -> ENPriors:
    # Fit on standardized X for numerical stability.
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    en_cfg = cfg["elastic_net"]
    alpha = float(en_cfg["lambda_"])           # sklearn uses 'alpha' for lambda
    l1_ratio = float(en_cfg["alpha"])
    eps = float(en_cfg.get("epsilon", 0.0))

    # ElasticNet is regression; for classification labels we use y as {0,1} / {-1,1}.
    yv = y.astype(float)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=int(cfg["seed"]))
    model.fit(Xs, yv)

    beta = model.coef_.astype(float)
    a = np.abs(beta)

    if float(a.sum()) == 0.0:
        a = np.ones_like(a)

    a = a + eps
    w = a / a.sum()
    ranking = np.argsort(-w)
    return ENPriors(w=w, ranking=ranking)
