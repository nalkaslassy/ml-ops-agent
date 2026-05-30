"""
xgboost_clf.py — XGBoost classifier for sentiment analysis.

Defines the algorithm and its hyperparameter search space.
Used by SentimentModel to compete against other algorithms.
"""

from xgboost import XGBClassifier


ALGORITHM_NAME = "XGBoost"


def get_estimator():
    """Return a fresh XGBClassifier instance."""
    return XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def get_param_grid():
    """
    Return the hyperparameter search space for RandomizedSearchCV.
    Keys must be prefixed with 'classifier__' to target the pipeline step.
    """
    return {
        "classifier__n_estimators":  [100, 200, 300],
        "classifier__max_depth":     [3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample":     [0.7, 0.9, 1.0],
    }
