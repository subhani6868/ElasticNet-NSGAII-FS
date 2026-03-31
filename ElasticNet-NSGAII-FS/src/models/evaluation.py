from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.models.classifiers import make_classifier

@dataclass
class EvalResult:
    mean_acc: float
    std_acc: float

def evaluate_subset_cv(X: np.ndarray, y: np.ndarray, s: np.ndarray, cfg: dict) -> EvalResult:
    idx = np.where(s.astype(bool))[0]
    if idx.size == 0:
        return EvalResult(mean_acc=0.0, std_acc=1.0)

    Xs = X[:, idx]

    cv_cfg = cfg["cv"]
    rskf = RepeatedStratifiedKFold(
        n_splits=int(cv_cfg["n_splits"]),
        n_repeats=int(cv_cfg["n_repeats"]),
        random_state=int(cfg["seed"])
    )

    clf_name = cfg["evaluation"]["classifier"]
    accs = []

    for train_idx, test_idx in rskf.split(Xs, y):
        Xtr, Xte = Xs[train_idx], Xs[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        clf = make_classifier(clf_name, random_state=int(cfg["seed"]))
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        accs.append(accuracy_score(yte, yhat))

    accs = np.asarray(accs, dtype=float)
    return EvalResult(mean_acc=float(accs.mean()), std_acc=float(accs.std(ddof=1) if accs.size > 1 else 0.0))
