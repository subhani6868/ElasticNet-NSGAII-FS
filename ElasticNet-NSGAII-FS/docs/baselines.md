# Baselines implemented

- **nsga2_vanilla**: Standard NSGA-II without Elastic Net guidance (gamma=0), uniform priors.
- **elasticnet_topk**: Rank features by Elastic Net priors and choose top-k (k chosen by CV from k_grid).
- **lasso_logreg**: L1-penalized logistic regression; selected features are non-zero coefficients.
- **mi_topk**: Mutual information filter; choose top-k (k chosen by CV).
- **random_topk**: Random top-k (sanity check); k chosen by CV.

Outputs:
- `results/metrics/<tag>_baseline_metrics.csv`
- `results/metrics/<tag>_comparison.csv` (includes the proposed method row)

Configure in `configs/base.yaml` under `baselines:`.
