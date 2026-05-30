"""
Sentiment algorithm registry.
SentimentModel imports this to get all candidate algorithms dynamically.
Add a new algorithm by dropping a new file here and registering it below.
"""

from . import logistic_clf, random_forest_clf, xgboost_clf

ALGORITHM_REGISTRY = [
    logistic_clf,
    random_forest_clf,
    xgboost_clf,
]
