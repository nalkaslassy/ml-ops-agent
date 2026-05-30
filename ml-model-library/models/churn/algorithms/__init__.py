"""
Churn algorithm registry.
ChurnModel imports this to get all candidate algorithms dynamically.
Add a new algorithm by dropping a new file in this folder and registering it here.
"""

from . import xgboost_clf, random_forest_clf, logistic_clf

# Registry — each entry is a module with get_estimator() and get_param_grid()
ALGORITHM_REGISTRY = [
    xgboost_clf,
    random_forest_clf,
    logistic_clf,
]
