"""
FraudModel — coming soon.

Planned:
- Combines Isolation Forest (unsupervised) + Random Forest (supervised)
- Uses Precision-Recall AUC as the metric (better than ROC-AUC for extreme class imbalance)
- Compatible data: transaction datasets where fraud is typically <1% of records

Usage (once implemented):
    from fraud.fraud_model import FraudModel

    model = FraudModel(target_col="is_fraud", drop_cols=["transaction_id"])
    model.fit(df)
    predictions = model.predict(new_df)
"""


class FraudModel:

    def __init__(self, target_col="is_fraud", drop_cols=None):
        raise NotImplementedError(
            "FraudModel is not yet implemented. "
            "Coming soon: Isolation Forest + Random Forest with Precision-Recall AUC."
        )

    def fit(self, df):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError
