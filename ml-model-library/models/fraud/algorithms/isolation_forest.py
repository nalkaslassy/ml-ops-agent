"""
isolation_forest.py — Isolation Forest for fraud detection.

TODO: Implement.

Isolation Forest is an unsupervised anomaly detection algorithm.
It isolates observations by randomly selecting a feature and
a split value — anomalies (fraud) require fewer splits to isolate.

Planned interface:
    ALGORITHM_NAME = "Isolation Forest"

    def get_estimator(): ...
    def get_param_grid(): ...  # n_estimators, contamination, max_samples
"""

ALGORITHM_NAME = "Isolation Forest"


def get_estimator():
    # TODO
    raise NotImplementedError


def get_param_grid():
    # TODO
    raise NotImplementedError
