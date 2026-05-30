"""
Fraud detection algorithm registry.
"""

from . import isolation_forest, random_forest_clf

ALGORITHM_REGISTRY = [
    isolation_forest,
    random_forest_clf,
]
