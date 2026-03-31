from __future__ import annotations
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def make_classifier(name: str, random_state: int) -> Any:
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(
            solver="saga",
            penalty="l2",
            max_iter=5000,
            n_jobs=None,
            random_state=random_state
        )
    if name == "linear_svm":
        return LinearSVC(random_state=random_state)
    raise ValueError(f"Unknown classifier: {name}")
