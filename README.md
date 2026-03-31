# Elastic Net–Guided NSGA-II for Multi-Objective Feature Selection (EN-NSGAII-FS)

This repository provides a reproducible reference implementation of the methodology described in the manuscript:
**"Elastic Net–Guided NSGA-II for Scalable Multi-Objective Feature Selection in High-Dimensional Data Analytics"**.

## What this code implements
- **Algorithm 1**: Elastic Net prior extraction and weight normalization (feature priors `w`)
- **Algorithm 2**: Elastic Net–guided NSGA-II (guided initialization + guided mutation)
- **Algorithm 3**: Stability-aware knee-point selection from the Pareto set

Objectives (minimized):
- `f1(s)`: classification error = `1 - mean_cv_accuracy`
- `f2(s)`: subset size ratio = `|s|/d`
- `f3(s)`: instability proxy = `std_cv_accuracy` (lower std ⇒ higher stability)

> Note: If you prefer an alternative stability definition (e.g., Kuncheva/Jaccard across repeated selections),
> you can switch `f3` in `src/feature_selection/objectives.py`.

## Datasets
- Core benchmarks (B1–B5): ARCENE, DEXTER, DOROTHEA, GISETTE, MADELON (UCI NIPS 2003 feature selection challenge pages).
  Access can vary by dataset (some require manual download). You can:
  1) place files under `data/raw/<dataset_name>/`, or
  2) use OpenML mirrors when available.

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# 1) Put dataset files under data/raw/... (see docs/dataset_instructions.md)
# 2) Prepare processed cache
python scripts/prepare_data.py --dataset madelon

# Run one experiment
python -m src.main --config configs/experiments/b5_madelon.yaml
```

## Outputs
Results are written under `results/`:
- `pareto_fronts/*.csv` : objective values + chromosome summary
- `selected_subsets/*.json` : final selected subset indices
- `metrics/*.csv` : summary metrics across runs
- `figures/*.png` : Pareto scatter + knee-point mark (if enabled)

## Reproducibility
- Fixed seeds via config
- Leakage-safe preprocessing (fit scalers on train fold only)
- Identical splits across methods (CV generator)

## License
This project is released under the MIT License.
If you use this code in your research, please cite the associated work.


## Baselines
Baselines are implemented in `src/experiments/baselines.py`. See `docs/baselines.md`.
