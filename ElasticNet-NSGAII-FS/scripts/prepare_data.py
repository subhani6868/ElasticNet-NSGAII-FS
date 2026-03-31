from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from src.datasets.preprocess import load_prepared_or_prepare

def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset key (e.g., madelon)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_cfg = load_yaml(root / "configs" / "base.yaml")
    datasets_cfg = load_yaml(root / "configs" / "datasets.yaml")
    cfg = dict(base_cfg)
    cfg["datasets"] = datasets_cfg["datasets"]

    load_prepared_or_prepare(args.dataset, root, cfg)
    print(f"Prepared: {args.dataset}")

if __name__ == "__main__":
    main()
