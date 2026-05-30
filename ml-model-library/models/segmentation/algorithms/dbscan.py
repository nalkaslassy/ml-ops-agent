"""
dbscan.py — DBSCAN clustering for customer segmentation.

TODO: Implement.

DBSCAN is density-based — it finds clusters of arbitrary shape
and flags outliers as noise (-1 label). Useful when customer
segments aren't spherical (unlike K-Means assumption).

Planned interface:
    ALGORITHM_NAME = "DBSCAN"

    def get_estimator(): ...
    def get_param_grid(): ...  # will cover eps, min_samples
"""

ALGORITHM_NAME = "DBSCAN"


def get_estimator():
    # TODO
    raise NotImplementedError


def get_param_grid():
    # TODO
    raise NotImplementedError
