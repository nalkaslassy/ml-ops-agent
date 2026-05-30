"""
churn_model.py — Auto-adaptive customer churn prediction.

Orchestrates the algorithm registry in algorithms/.
Each algorithm lives in its own file — add new ones by dropping a module
in algorithms/ and registering it in algorithms/__init__.py.

Usage:
    from models.churn.churn_model import ChurnModel

    model = ChurnModel(target_col='churned', drop_cols=['customer_id'])
    model.fit(df)
    predictions = model.predict(new_df)
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from .algorithms import ALGORITHM_REGISTRY

warnings.filterwarnings("ignore")


class ChurnModel:
    """
    Auto-adaptive churn prediction model.

    Loads all algorithm modules from algorithms/,
    trains each one, tunes hyperparameters, and selects the best performer.

    Parameters
    ----------
    target_col   : str        — Column that indicates churn (binary 0/1).
    drop_cols    : list[str]  — Columns to exclude (IDs, dates, etc.).
    test_size    : float      — Held-out test fraction. Default 0.2.
    random_state : int        — Reproducibility seed. Default 42.
    n_iter       : int        — Hyperparameter combinations per algorithm. Default 10.
    cv           : int        — Cross-validation folds. Default 5.
    """

    def __init__(
        self,
        target_col="churned",
        drop_cols=None,
        test_size=0.2,
        random_state=42,
        n_iter=10,
        cv=5,
    ):
        self.target_col   = target_col
        self.drop_cols    = drop_cols or []
        self.test_size    = test_size
        self.random_state = random_state
        self.n_iter       = n_iter
        self.cv           = cv

        self.best_model      = None
        self.best_model_name = None
        self.feature_names_  = None
        self.is_fitted       = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_preprocessor(self, X):
        """Auto-detect column types and build ColumnTransformer."""
        numeric_cols     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        print(f"  Numeric    ({len(numeric_cols)}): {numeric_cols}")
        print(f"  Categorical({len(categorical_cols)}): {categorical_cols}")

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if numeric_cols:     transformers.append(("num", num_pipe, numeric_cols))
        if categorical_cols: transformers.append(("cat", cat_pipe, categorical_cols))

        if not transformers:
            raise ValueError("No usable features found after dropping target and drop_cols.")

        return ColumnTransformer(transformers=transformers)

    def _load_candidates(self):
        """
        Pull algorithm name, estimator, and param grid from every module
        in the algorithm registry.
        """
        candidates = {}
        for module in ALGORITHM_REGISTRY:
            name       = module.ALGORITHM_NAME
            estimator  = module.get_estimator()
            param_grid = module.get_param_grid()
            candidates[name] = (estimator, param_grid)
        return candidates

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(self, df):
        """
        Train the model. Automatically selects the best algorithm for your data.

        Parameters
        ----------
        df : pd.DataFrame — Must contain target_col. All other columns
             (except drop_cols) are treated as features.

        Returns self.
        """
        print("=" * 55)
        print("  ChurnModel — Training")
        print("=" * 55)

        if self.target_col not in df.columns:
            raise ValueError(
                f"target_col '{self.target_col}' not in DataFrame. "
                f"Available: {df.columns.tolist()}"
            )

        cols_to_drop = [self.target_col] + [c for c in self.drop_cols if c in df.columns]
        X = df.drop(columns=cols_to_drop)
        y = df[self.target_col]
        self.feature_names_ = X.columns.tolist()

        churn_rate = y.mean()
        print(f"\nRows: {len(X):,}  |  Features: {len(X.columns)}  |  Churn rate: {churn_rate:.1%}")

        if churn_rate < 0.05 or churn_rate > 0.95:
            print("  ⚠  Class imbalance detected — consider SMOTE for better minority recall.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        print("\nDetecting features:")
        preprocessor = self._build_preprocessor(X_train)
        candidates   = self._load_candidates()

        print(f"\nTesting {len(candidates)} algorithms "
              f"({self.n_iter} configs × {self.cv}-fold CV each)...\n")

        best_score = -1
        cv_results = {}

        for name, (estimator, param_grid) in candidates.items():
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier",   estimator),
            ])
            search = RandomizedSearchCV(
                pipeline, param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring="roc_auc",
                random_state=self.random_state,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            score = search.best_score_
            cv_results[name] = score

            marker = "  ← best so far" if score > best_score else ""
            print(f"  {name:<25}  CV AUC = {score:.4f}{marker}")

            if score > best_score:
                best_score           = score
                self.best_model      = search.best_estimator_
                self.best_model_name = name
                self._best_params    = search.best_params_

        print(f"\n✅ Winner: {self.best_model_name}  (CV AUC: {best_score:.4f})")
        print(f"   Params:  {self._best_params}")

        # Evaluate on held-out test set
        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)

        print(f"\nTest AUC: {test_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

        self._plot_results(y_test, y_pred, y_prob, cv_results)

        self._X_test = X_test
        self._y_test = y_test
        self._y_prob = y_prob
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Public: predict
    # ------------------------------------------------------------------

    def predict(self, df):
        """
        Predict churn for new customers.

        Returns pd.DataFrame with:
            churn_probability : float [0, 1]
            will_churn        : int   {0, 1}
            risk_tier         : str   Low / Medium / High
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        X = df.drop(columns=[c for c in [self.target_col] + self.drop_cols if c in df.columns])
        probs = self.best_model.predict_proba(X)[:, 1]
        preds = self.best_model.predict(X)

        risk_tiers = pd.cut(
            probs,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )

        return pd.DataFrame({
            "churn_probability": np.round(probs, 4),
            "will_churn":        preds,
            "risk_tier":         risk_tiers,
        }, index=df.index)

    # ------------------------------------------------------------------
    # Public: feature_importance
    # ------------------------------------------------------------------

    def feature_importance(self, top_n=15):
        """Plot feature importances (tree-based models only)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before feature_importance().")

        clf = self.best_model.named_steps["classifier"]
        if not hasattr(clf, "feature_importances_"):
            print(f"{self.best_model_name} doesn't expose feature importances.")
            return

        preprocessor = self.best_model.named_steps["preprocessor"]
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]

        names = [re.sub(r"^(num__|cat__)", "", n) for n in names]
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            [names[i] for i in reversed(idx)],
            [importances[i] for i in reversed(idx)],
            color="#3498db",
        )
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top {top_n} Feature Importances — {self.best_model_name}")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Private: plotting
    # ------------------------------------------------------------------

    def _plot_results(self, y_test, y_pred, y_prob, cv_results):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Churn Model Results  |  Winner: {self.best_model_name}",
            fontsize=14, fontweight="bold",
        )

        max_score = max(cv_results.values())
        colors    = ["#2ecc71" if v == max_score else "#bdc3c7" for v in cv_results.values()]

        bars = axes[0].barh(list(cv_results.keys()), list(cv_results.values()), color=colors)
        axes[0].set_xlabel("CV ROC-AUC Score")
        axes[0].set_title("Algorithm Comparison")
        axes[0].set_xlim(0, 1.08)
        for bar, v in zip(bars, cv_results.values()):
            axes[0].text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                         f"{v:.4f}", va="center", fontsize=10)

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Retained", "Churned"],
            ax=axes[1], colorbar=False, cmap="Blues",
        )
        axes[1].set_title("Confusion Matrix (Test Set)")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        axes[2].plot(fpr, tpr, color="#2ecc71", lw=2, label=f"AUC = {auc:.4f}")
        axes[2].fill_between(fpr, tpr, alpha=0.1, color="#2ecc71")
        axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
        axes[2].set_xlabel("False Positive Rate")
        axes[2].set_ylabel("True Positive Rate")
        axes[2].set_title("ROC Curve (Test Set)")
        axes[2].legend(loc="lower right")

        plt.tight_layout()
        plt.show()
