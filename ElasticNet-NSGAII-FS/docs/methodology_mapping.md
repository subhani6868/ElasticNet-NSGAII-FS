# Methodology ↔ Code mapping

- Algorithm 1 → src/feature_selection/elastic_net.py
- Algorithm 2 → src/feature_selection/nsga2_core.py + src/feature_selection/operators.py
- Algorithm 3 → src/feature_selection/pareto_select.py

Equations:
- (1) Elastic Net objective → sklearn ElasticNet (see elastic_net.py)
- (5)-(7) objectives → src/feature_selection/objectives.py
- (11)-(13) Pareto normalization/utility/knee → src/feature_selection/pareto_select.py
