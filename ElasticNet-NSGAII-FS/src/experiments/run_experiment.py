from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.datasets.registry import load_dataset
from src.feature_selection.elastic_net import elastic_net_priors
from src.feature_selection.nsga2_core import nsga2_guided
from src.feature_selection.pareto_select import select_final_subset
from src.utils.io import ensure_dir, save_json
from src.reporting.plots import plot_pareto
from src.experiments.baselines import run_baselines, baselines_to_frame

def run_experiment(cfg: dict, root: Path) -> None:
    dataset_key = cfg["dataset"]
    tag = cfg.get("output_tag", dataset_key)

    X, y, feature_names = load_dataset(dataset_key, root, cfg)
    d = X.shape[1]

    # Algorithm 1
    pri = elastic_net_priors(X, y, cfg)
    w = pri.w

    # Algorithm 2
    pareto_inds = nsga2_guided(X, y, w, cfg)
    pareto_s = [ind.s for ind in pareto_inds]
    pareto_F = np.asarray([ind.f for ind in pareto_inds], dtype=float)

    # Algorithm 3
    sel_cfg = cfg["selection"]
    omega = tuple(sel_cfg["omega"])
    top_k = int(sel_cfg["knee_top_k"])
    decision = select_final_subset(pareto_s, pareto_F, omega, top_k)
    s_star = decision.s_star

    # Save results
    out_dir = root / "results"
    ensure_dir(out_dir / "pareto_fronts")
    ensure_dir(out_dir / "selected_subsets")
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "figures")

    # Pareto CSV
    df = pd.DataFrame(pareto_F, columns=["f1_error", "f2_size_ratio", "f3_instability"])
    df["n_features"] = [int(np.count_nonzero(s)) for s in pareto_s]
    df.to_csv(out_dir / "pareto_fronts" / f"{tag}_pareto.csv", index=False)

    # Selected subset JSON
    idx = np.where(s_star.astype(bool))[0].tolist()
    payload = {
        "dataset": dataset_key,
        "tag": tag,
        "selected_indices": idx,
        "n_features": int(len(idx)),
        "feature_names": [feature_names[i] for i in idx] if feature_names is not None else None,
        "chosen_pareto_index": int(decision.chosen_index),
    }
    save_json(payload, out_dir / "selected_subsets" / f"{tag}_selected.json")

    # Simple metrics summary for selected subset (re-evaluate)
    from src.feature_selection.objectives import objectives
    f1, f2, f3 = objectives(X, y, s_star, cfg)
    mdf = pd.DataFrame([{
        "dataset": dataset_key,
        "tag": tag,
        "d": int(d),
        "n_selected": int(np.count_nonzero(s_star)),
        "f1_error": float(f1),
        "f2_size_ratio": float(f2),
        "f3_instability": float(f3),
        "mean_accuracy": float(1.0 - f1),
    }])
    # Baselines (optional)
bcfg = cfg.get("baselines", {}) or {}
if bool(bcfg.get("enabled", False)):
    bres = run_baselines(X, y, cfg)
    bdf = baselines_to_frame(bres, d=int(d))
    bdf.to_csv(out_dir / "metrics" / f"{tag}_baseline_metrics.csv", index=False)

    proposed_row = pd.DataFrame([{
        "method": "proposed_en_guided_nsga2",
        "d": int(d),
        "n_selected": int(np.count_nonzero(s_star)),
        "chosen_k": int(np.count_nonzero(s_star)),
        "f1_error": float(f1),
        "f2_size_ratio": float(f2),
        "f3_instability": float(f3),
        "mean_accuracy": float(1.0 - f1),
    }])
    comp = pd.concat([proposed_row, bdf], ignore_index=True)
    comp.to_csv(out_dir / "metrics" / f"{tag}_comparison.csv", index=False)

    # Plot
    if bool(cfg.get("include_plots", True)):
        plot_pareto(pareto_F, decision, out_dir / "figures" / f"{tag}_pareto.png")
