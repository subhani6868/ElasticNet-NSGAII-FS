import argparse
from pathlib import Path
import yaml

from src.experiments.run_experiment import run_experiment
from src.utils.seed import set_global_seed
from src.utils.logging import get_logger
from src.utils.io import deep_merge_dicts

def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment yaml under configs/experiments/")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_cfg = load_yaml(root / "configs" / "base.yaml")
    datasets_cfg = load_yaml(root / "configs" / "datasets.yaml")
    exp_cfg = load_yaml(Path(args.config))

    cfg = deep_merge_dicts(base_cfg, exp_cfg)
    cfg["datasets"] = datasets_cfg["datasets"]

    set_global_seed(cfg["seed"])
    log = get_logger()
    log.info("Running experiment with config: %s", cfg)

    run_experiment(cfg, root)

if __name__ == "__main__":
    main()
