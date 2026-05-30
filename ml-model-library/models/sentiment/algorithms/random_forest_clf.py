"""
random_forest_clf.py — Random Forest classifier for sentiment analysis.

Defines the algorithm and its hyperparameter search space.
Used by SentimentModel to compete against other algorithms.
"""

from sklearn.ensemble import RandomForestClassifier


ALGORITHM_NAME = "Random Forest"


def get_estimator():
    """Return a fresh RandomForestClassifier instance."""
    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )


def get_param_grid():
    """
    Return the hyperparameter search space for RandomizedSearchCV.
    Keys must be prefixed with 'classifier__' to target the pipeline step.
    """
    return {
        "classifier__n_estimators":     [100, 200, 300],
        "classifier__max_depth":        [None, 10, 20, 30],
        "classifier__min_samples_split":[2, 5, 10],
        "classifier__max_features":     ["sqrt", "log2"],
    }
