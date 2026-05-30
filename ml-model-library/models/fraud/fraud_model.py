"""
fraud_model.py — Auto-adaptive fraud detection.

TODO: Implement FraudModel.

Handles extreme class imbalance (fraud is rare — typically <1% of transactions).
Combines Isolation Forest (unsupervised) with Random Forest (supervised).
Evaluates using precision-recall AUC, not ROC AUC (more appropriate for imbalanced data).

Planned usage:
    from models.fraud.fraud_model import FraudModel

    model = FraudModel(target_col='is_fraud', drop_cols=['transaction_id'])
    model.fit(df)
    predictions = model.predict(new_transactions)

Algorithm modules ready in algorithms/:
    - isolation_forest.py
    - random_forest_clf.py
"""


class FraudModel:
    """
    Auto-adaptive fraud detection model.
    TODO: Implement.
    """

    def __init__(self, target_col="is_fraud", drop_cols=None):
        self.target_col = target_col
        self.drop_cols  = drop_cols or []
        self.is_fitted  = False
        raise NotImplementedError("FraudModel is not yet implemented.")

    def fit(self, df):
        # TODO
        raise NotImplementedError

    def predict(self, df):
        # TODO
        raise NotImplementedError
