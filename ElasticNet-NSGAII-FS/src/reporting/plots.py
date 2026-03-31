from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_pareto(F: np.ndarray, decision, out_path: Path) -> None:
    # Plot f1 vs f2; mark chosen index
    F = np.asarray(F, dtype=float)
    x = F[:, 1]  # size ratio
    y = F[:, 0]  # error
    plt.figure()
    plt.scatter(x, y)
    ci = int(decision.chosen_index)
    plt.scatter([x[ci]], [y[ci]], marker="x")
    plt.xlabel("f2: Subset Size Ratio")
    plt.ylabel("f1: Classification Error")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
