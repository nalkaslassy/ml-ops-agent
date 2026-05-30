"""
SegmentationModel — coming soon.

Planned:
- Unsupervised clustering (no target label needed)
- Tests K-Means and DBSCAN
- Auto-selects optimal number of clusters via silhouette score
- Returns cluster labels + segment profiles for targeting

Usage (once implemented):
    from segmentation.segmentation_model import SegmentationModel

    model = SegmentationModel(drop_cols=["customer_id"])
    model.fit(df)
    segments = model.predict(new_df)
"""


class SegmentationModel:

    def __init__(self, drop_cols=None, n_clusters=None):
        raise NotImplementedError(
            "SegmentationModel is not yet implemented. "
            "Coming soon: K-Means + DBSCAN with silhouette score selection."
        )

    def fit(self, df):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError
