"""
segmentation_model.py — Auto-adaptive customer segmentation.

TODO: Implement SegmentationModel.

Unsupervised — no target column needed.
Auto-scales features, tests K-Means and DBSCAN,
auto-selects optimal number of clusters (silhouette score),
and returns cluster labels + segment profiles.

Planned usage:
    from models.segmentation.segmentation_model import SegmentationModel

    model = SegmentationModel(drop_cols=['customer_id'])
    model.fit(df)
    segments = model.predict(df)

Algorithm modules ready in algorithms/:
    - kmeans.py
    - dbscan.py
"""


class SegmentationModel:
    """
    Auto-adaptive customer segmentation model.
    TODO: Implement.
    """

    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols or []
        self.is_fitted = False
        raise NotImplementedError("SegmentationModel is not yet implemented.")

    def fit(self, df):
        # TODO
        raise NotImplementedError

    def predict(self, df):
        # TODO
        raise NotImplementedError
