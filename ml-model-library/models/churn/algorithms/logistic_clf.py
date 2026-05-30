"""
logistic_clf.py — Logistic Regression classifier for churn prediction.

Defines the algorithm and its hyperparameter search space.
Used by ChurnModel to compete against other algorithms.
"""

from sklearn.linear_model import LogisticRegression


ALGORITHM_NAME = "Logistic Regression"


def get_estimator():
    """Return a fresh LogisticRegression instance."""
    return LogisticRegression(
        random_state=42,
        max_iter=1000,
    )


def get_param_grid():
    """
    Return the hyperparameter search space for RandomizedSearchCV.
    Keys must be prefixed with 'classifier__' to target the pipeline step.
    """
    return {
        "classifier__C":       [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__solver":  ["lbfgs", "saga"],
        "classifier__penalty": ["l2"],
    }
