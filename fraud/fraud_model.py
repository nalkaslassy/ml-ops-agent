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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, average_precision_score,
    precision_recall_curve, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier


class FraudModel:
    """
    Detects fraudulent transactions. Built for datasets where fraud is a small
    fraction of records (e.g. <1%). Uses Precision-Recall AUC as the scoring
    metric — more meaningful than ROC-AUC when classes are heavily imbalanced.

    All 3 algorithms are configured to handle class imbalance automatically.
    """

    def __init__(self, target_col="is_fraud", drop_cols=None, test_size=0.2,
                 random_state=42, n_iter=10, cv=5):
        self.target_col   = target_col
        self.drop_cols    = drop_cols or []
        self.test_size    = test_size
        self.random_state = random_state
        self.n_iter       = n_iter
        self.cv           = cv
        self.best_model   = None
        self.best_name    = None
        self.best_score   = None
        self.is_fitted    = False

    def _validate(self, df):
        if len(df) < 200:
            raise ValueError(
                f"Dataset has only {len(df)} rows — at least 200 are required to train reliably."
            )
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        unique_vals = set(df[self.target_col].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Target column '{self.target_col}' must be binary (0 = legitimate, 1 = fraud), "
                f"but found: {sorted(unique_vals)}"
            )
        drop = [self.target_col] + [c for c in self.drop_cols if c in df.columns]
        feature_cols = [c for c in df.columns if c not in drop]
        if len(feature_cols) < 2:
            raise ValueError(
                f"At least 2 feature columns required, but only {len(feature_cols)} remain: {feature_cols}"
            )

    def _build_algorithms(self, scale_pos_weight):
        return [
            {
                "name": "XGBoost",
                "model": XGBClassifier(
                    eval_metric="aucpr",
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
                "params": {
                    "classifier__n_estimators":     [100, 200, 300, 500],
                    "classifier__max_depth":        [3, 4, 5, 6],
                    "classifier__learning_rate":    [0.01, 0.05, 0.1, 0.2],
                    "classifier__subsample":        [0.7, 0.8, 0.9, 1.0],
                    "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "classifier__min_child_weight": [1, 3, 5],
                }
            },
            {
                "name": "Random Forest",
                "model": RandomForestClassifier(
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
                "params": {
                    "classifier__n_estimators":      [100, 200, 300],
                    "classifier__max_depth":         [None, 5, 10, 20],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__min_samples_leaf":  [1, 2, 4],
                    "classifier__max_features":      ["sqrt", "log2"],
                }
            },
            {
                "name": "Logistic Regression",
                "model": LogisticRegression(
                    class_weight="balanced",
                    random_state=self.random_state,
                    max_iter=1000,
                ),
                "params": {
                    "classifier__C":       [0.01, 0.1, 1, 10, 100],
                    "classifier__solver":  ["lbfgs", "saga"],
                    "classifier__penalty": ["l2"],
                }
            },
        ]

    def _build_preprocessor(self, X):
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        print(f"  Numeric columns ({len(num_cols)}):     {num_cols}")
        print(f"  Categorical columns ({len(cat_cols)}): {cat_cols}\n")
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers = []
        if num_cols: transformers.append(("num", num_pipe, num_cols))
        if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
        return ColumnTransformer(transformers)

    def fit(self, df):
        self._validate(df)
        drop = [self.target_col] + [c for c in self.drop_cols if c in df.columns]
        X = df.drop(columns=drop)
        y = df[self.target_col]

        fraud_rate     = y.mean()
        n_fraud        = int(y.sum())
        n_legit        = int((y == 0).sum())
        scale_pos_weight = n_legit / n_fraud

        print(f"Training on {len(X):,} rows | {len(X.columns)} features | Fraud rate: {fraud_rate:.2%}")
        print(f"  Legitimate: {n_legit:,} | Fraud: {n_fraud:,} | Class ratio: {scale_pos_weight:.1f}:1\n")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        preprocessor = self._build_preprocessor(X_train)
        algorithms   = self._build_algorithms(scale_pos_weight)
        best_score   = -1
        results      = {}

        print(f"Testing {len(algorithms)} algorithms ({self.n_iter} configs x {self.cv}-fold CV each)...")
        print("Metric: Average Precision (Precision-Recall AUC)\n")

        for algo in algorithms:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier",   algo["model"]),
            ])
            search = RandomizedSearchCV(
                pipeline, algo["params"],
                n_iter=self.n_iter, cv=self.cv, scoring="average_precision",
                random_state=self.random_state, n_jobs=-1
            )
            search.fit(X_train, y_train)
            score = search.best_score_
            results[algo["name"]] = score
            marker = "  <- best so far" if score > best_score else ""
            print(f"  {algo['name']:<25} Avg Precision = {score:.4f}{marker}")

            if score > best_score:
                best_score        = score
                self.best_model   = search.best_estimator_
                self.best_name    = algo["name"]
                self._best_params = search.best_params_

        self.best_score = best_score
        print(f"\nWinner: {self.best_name} (Avg Precision: {best_score:.4f})")
        print(f"Params: {self._best_params}\n")

        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        print(f"Test Average Precision: {average_precision_score(y_test, y_prob):.4f}\n")
        print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
        self._plot(y_test, y_pred, y_prob, results)
        self.is_fitted = True
        return self

    def predict(self, df):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X     = df.drop(columns=[c for c in [self.target_col] + self.drop_cols if c in df.columns])
        probs = self.best_model.predict_proba(X)[:, 1]
        preds = self.best_model.predict(X)
        return pd.DataFrame({
            "fraud_probability": np.round(probs, 4),
            "is_fraud":          preds,
        }, index=df.index)

    def _plot(self, y_test, y_pred, y_prob, results):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Fraud Model — Winner: {self.best_name}", fontsize=13, fontweight="bold")

        best   = max(results.values())
        colors = ["#e67e22" if v == best else "#cccccc" for v in results.values()]
        bars   = axes[0].barh(list(results.keys()), list(results.values()), color=colors)
        axes[0].set_xlim(0, 1.1)
        axes[0].set_title("Algorithm Comparison (Avg Precision)")
        for bar, v in zip(bars, results.values()):
            axes[0].text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center")

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["Legitimate", "Fraud"],
            ax=axes[1], colorbar=False, cmap="Oranges"
        )
        axes[1].set_title("Confusion Matrix")

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        axes[2].plot(recall, precision, color="#e67e22", lw=2, label=f"AP = {ap:.4f}")
        axes[2].fill_between(recall, precision, alpha=0.1, color="#e67e22")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title("Precision-Recall Curve")
        axes[2].legend()

        plt.tight_layout()
        plt.show()
