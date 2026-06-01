import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer


KMEANS_K_VALUES = [2, 3, 4, 5, 6, 7, 8]

DBSCAN_PARAMS = [
    {"eps": 0.3, "min_samples": 5},
    {"eps": 0.5, "min_samples": 5},
    {"eps": 0.5, "min_samples": 10},
    {"eps": 1.0, "min_samples": 5},
    {"eps": 1.0, "min_samples": 10},
    {"eps": 1.5, "min_samples": 5},
    {"eps": 2.0, "min_samples": 5},
]


class SegmentationModel:
    """
    Groups customers into segments automatically. No target label needed.

    Tests K-Means (k=2 through 8) and DBSCAN across multiple settings.
    Picks the configuration with the highest silhouette score — a measure
    of how well-separated the clusters are.

    Only works with numeric columns. Drop ID columns using drop_cols.
    """

    def __init__(self, drop_cols=None, random_state=42):
        self.drop_cols    = drop_cols or []
        self.random_state = random_state
        self.scaler       = None
        self.best_model   = None
        self.best_name    = None
        self.best_score   = None
        self.labels_      = None
        self.is_fitted    = False

    def _validate(self, df):
        if len(df) < 50:
            raise ValueError(
                f"Dataset has only {len(df)} rows — at least 50 are needed for clustering."
            )
        drop = [c for c in self.drop_cols if c in df.columns]
        feature_cols = df.drop(columns=drop).select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(feature_cols) < 2:
            raise ValueError(
                f"At least 2 numeric feature columns are required for clustering, "
                f"but only {len(feature_cols)} found: {feature_cols}"
            )

    def fit(self, df):
        self._validate(df)
        drop  = [c for c in self.drop_cols if c in df.columns]
        X_raw = df.drop(columns=drop).select_dtypes(include=["int64", "float64"])

        self._feature_cols = X_raw.columns.tolist()
        print(f"Clustering {len(X_raw):,} rows | {len(X_raw.columns)} features")
        print(f"Features: {self._feature_cols}\n")

        self._imputer = SimpleImputer(strategy="median")
        self.scaler   = StandardScaler()
        X = self.scaler.fit_transform(self._imputer.fit_transform(X_raw))

        best_score = -1
        results    = {}

        print(f"Testing K-Means (k = {KMEANS_K_VALUES[0]} to {KMEANS_K_VALUES[-1]})...")
        for k in KMEANS_K_VALUES:
            model  = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X)
            score  = silhouette_score(X, labels)
            name   = f"K-Means (k={k})"
            results[name] = score
            marker = "  <- best so far" if score > best_score else ""
            print(f"  k={k}  Silhouette = {score:.4f}{marker}")

            if score > best_score:
                best_score      = score
                self.best_model = model
                self.best_name  = name

        print("\nTesting DBSCAN...")
        for params in DBSCAN_PARAMS:
            model  = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
            labels = model.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters < 2:
                print(f"  eps={params['eps']} min_samples={params['min_samples']}  -> {n_clusters} cluster(s), skipping")
                continue

            valid_mask  = labels != -1
            noise_pct   = (~valid_mask).mean()
            score = silhouette_score(X[valid_mask], labels[valid_mask])
            name  = f"DBSCAN (eps={params['eps']}, min={params['min_samples']})"
            results[name] = score
            marker = "  <- best so far" if score > best_score else ""
            print(f"  eps={params['eps']} min={params['min_samples']}  Silhouette = {score:.4f} | {n_clusters} clusters | {noise_pct:.1%} noise{marker}")

            if score > best_score:
                best_score      = score
                self.best_model = model
                self.best_name  = name

        self.labels_   = self.best_model.labels_
        self._X_scaled = X
        self._X_raw    = X_raw.reset_index(drop=True)

        self.best_score = best_score
        print(f"\nWinner: {self.best_name} (Silhouette: {best_score:.4f})\n")
        self._print_profiles()
        self._plot(X, results)
        self.is_fitted = True
        return self

    def predict(self, df):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        drop  = [c for c in self.drop_cols if c in df.columns]
        X_raw = df.drop(columns=drop)[self._feature_cols]
        X     = self.scaler.transform(self._imputer.transform(X_raw))

        if hasattr(self.best_model, "predict"):
            labels = self.best_model.predict(X)
        else:
            # DBSCAN has no predict — assign to nearest cluster centroid
            from sklearn.metrics import pairwise_distances_argmin
            unique_labels = [l for l in np.unique(self.labels_) if l != -1]
            centroids = np.array([
                self._X_scaled[self.labels_ == l].mean(axis=0) for l in unique_labels
            ])
            labels = pairwise_distances_argmin(X, centroids)

        return pd.DataFrame({"segment": labels}, index=df.index)

    def segment_profiles(self):
        """Returns a summary table of each segment — mean value per feature."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before segment_profiles().")
        df = self._X_raw.copy()
        df["segment"] = self.labels_
        return df[df["segment"] != -1].groupby("segment").mean().round(2)

    def _print_profiles(self):
        df = self._X_raw.copy()
        df["segment"] = self.labels_
        valid = df[df["segment"] != -1]
        print(f"Segment profiles ({valid['segment'].nunique()} segments):\n")
        for seg in sorted(valid["segment"].unique()):
            group = valid[valid["segment"] == seg]
            print(f"  Segment {seg}  ({len(group):,} rows):")
            for col in self._feature_cols:
                print(f"    {col}: {group[col].mean():.2f}")
            print()

    def _plot(self, X, results):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Segmentation Model — Winner: {self.best_name}", fontsize=13, fontweight="bold")

        best   = max(results.values())
        colors = ["#9b59b6" if v == best else "#cccccc" for v in results.values()]
        bars   = axes[0].barh(list(results.keys()), list(results.values()), color=colors)
        axes[0].set_xlim(0, 1.1)
        axes[0].set_title("Algorithm Comparison (Silhouette Score)")
        for bar, v in zip(bars, results.values()):
            axes[0].text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center")

        pca  = PCA(n_components=2, random_state=self.random_state)
        X_2d = pca.fit_transform(X)
        unique_labels = sorted(set(self.labels_))
        palette = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, palette):
            mask = self.labels_ == label
            name = "Noise" if label == -1 else f"Segment {label} ({mask.sum():,})"
            axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=name, alpha=0.6, s=20)
        axes[1].set_title("Clusters (PCA — first 2 components)")
        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        axes[1].legend(markerscale=2, fontsize=8)

        plt.tight_layout()
        plt.show()
