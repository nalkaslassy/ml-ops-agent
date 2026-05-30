"""
ChurnModel — works with any churn dataset.

Just tell it which column is the churn label and which columns to ignore.
It handles everything else: feature detection, preprocessing, tuning, and picking the best algorithm.

Usage:
    from churn.churn_model import ChurnModel

    model = ChurnModel(target_col="churned", drop_cols=["customer_id"])
    model.fit(df)
    predictions = model.predict(new_df)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay

from churn import xgboost_clf, random_forest_clf, logistic_clf

# All algorithms to test — add/remove here
ALGORITHMS = [xgboost_clf, random_forest_clf, logistic_clf]


class ChurnModel:

    def __init__(self, target_col="churned", drop_cols=None):
        self.target_col = target_col
        self.drop_cols  = drop_cols or []
        self.best_model = None
        self.best_name  = None
        self.is_fitted  = False

    def fit(self, df):
        print("Training ChurnModel...")

        # Split features and label
        drop = [self.target_col] + [c for c in self.drop_cols if c in df.columns]
        X = df.drop(columns=drop)
        y = df[self.target_col]

        print(f"  {len(X)} rows | {len(X.columns)} features | churn rate: {y.mean():.1%}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Build preprocessor based on column types
        preprocessor = self._build_preprocessor(X_train)

        # Test every algorithm, pick the best one
        best_score = -1
        results = {}

        for algo in ALGORITHMS:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier",   algo.get_estimator()),
            ])
            search = RandomizedSearchCV(
                pipeline, algo.get_params(),
                n_iter=10, cv=5, scoring="roc_auc",
                random_state=42, n_jobs=-1
            )
            search.fit(X_train, y_train)
            score = search.best_score_
            results[algo.NAME] = score
            print(f"  {algo.NAME:<25} CV AUC = {score:.4f}")

            if score > best_score:
                best_score      = score
                self.best_model = search.best_estimator_
                self.best_name  = algo.NAME

        print(f"\n  Winner: {self.best_name} (CV AUC: {best_score:.4f})")

        # Final evaluation on test set
        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]

        print(f"  Test AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

        self._plot(y_test, y_pred, y_prob, results)
        self.is_fitted = True
        return self

    def predict(self, df):
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        X     = df.drop(columns=[c for c in [self.target_col] + self.drop_cols if c in df.columns])
        probs = self.best_model.predict_proba(X)[:, 1]
        preds = self.best_model.predict(X)
        tiers = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
        return pd.DataFrame({
            "churn_probability": np.round(probs, 4),
            "will_churn":        preds,
            "risk_tier":         tiers,
        }, index=df.index)

    def _build_preprocessor(self, X):
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        print(f"  Numeric: {num_cols}")
        print(f"  Categorical: {cat_cols}")

        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),   ("scaler", StandardScaler())])
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

        transformers = []
        if num_cols: transformers.append(("num", num_pipe, num_cols))
        if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
        return ColumnTransformer(transformers)

    def _plot(self, y_test, y_pred, y_prob, results):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Churn Model — Winner: {self.best_name}", fontsize=14, fontweight="bold")

        # Algorithm comparison
        best = max(results.values())
        colors = ["#2ecc71" if v == best else "#bdc3c7" for v in results.values()]
        bars = axes[0].barh(list(results.keys()), list(results.values()), color=colors)
        axes[0].set_xlim(0, 1.1)
        axes[0].set_title("Algorithm Comparison")
        for bar, v in zip(bars, results.values()):
            axes[0].text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center")

        # Confusion matrix
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["Retained", "Churned"],
            ax=axes[1], colorbar=False, cmap="Blues"
        )
        axes[1].set_title("Confusion Matrix")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        axes[2].plot(fpr, tpr, color="#2ecc71", lw=2, label=f"AUC = {auc:.4f}")
        axes[2].plot([0, 1], [0, 1], "k--", lw=1)
        axes[2].set_xlabel("False Positive Rate")
        axes[2].set_ylabel("True Positive Rate")
        axes[2].set_title("ROC Curve")
        axes[2].legend()

        plt.tight_layout()
        plt.show()
