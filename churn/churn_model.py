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
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from xgboost import XGBClassifier


ALGORITHMS = [
    {
        "name": "XGBoost",
        "model": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
        "params": {
            "classifier__n_estimators":  [100, 200, 300],
            "classifier__max_depth":     [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
        }
    },
    {
        "name": "Random Forest",
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "classifier__n_estimators":      [100, 200, 300],
            "classifier__max_depth":         [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
        }
    },
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "classifier__C":      [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs", "saga"],
        }
    },
]


class ChurnModel:
    """
    Predicts customer churn. Works with any dataset.
    Set target_col to your churn label column and drop_cols to any ID columns.
    Automatically detects features, tunes each algorithm, and picks the best one.
    """

    def __init__(self, target_col="churned", drop_cols=None):
        self.target_col = target_col
        self.drop_cols  = drop_cols or []
        self.best_model = None
        self.best_name  = None

    def _validate(self, df):
        if len(df) < 200:
            raise ValueError(
                f"Dataset has only {len(df)} rows — at least 200 are required to train reliably."
            )
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        unique_vals = set(df[self.target_col].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Target column '{self.target_col}' must be binary (only 0s and 1s), "
                f"but found values: {sorted(unique_vals)}"
            )
        drop = [self.target_col] + [c for c in self.drop_cols if c in df.columns]
        feature_cols = [c for c in df.columns if c not in drop]
        if len(feature_cols) < 2:
            raise ValueError(
                f"At least 2 feature columns are required after dropping target and drop_cols, "
                f"but only {len(feature_cols)} remain: {feature_cols}"
            )

    def fit(self, df):
        self._validate(df)
        drop = [self.target_col] + [c for c in self.drop_cols if c in df.columns]
        X = df.drop(columns=drop)
        y = df[self.target_col]

        print(f"Training on {len(X)} rows, {len(X.columns)} features, churn rate: {y.mean():.1%}\n")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        preprocessor = self._build_preprocessor(X_train)
        best_score = -1
        results = {}

        for algo in ALGORITHMS:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier",   algo["model"]),
            ])
            search = RandomizedSearchCV(
                pipeline, algo["params"],
                n_iter=10, cv=5, scoring="roc_auc",
                random_state=42, n_jobs=-1
            )
            search.fit(X_train, y_train)
            score = search.best_score_
            results[algo["name"]] = score
            print(f"  {algo['name']:<25} AUC = {score:.4f}")

            if score > best_score:
                best_score      = score
                self.best_model = search.best_estimator_
                self.best_name  = algo["name"]

        print(f"\nWinner: {self.best_name} (AUC: {best_score:.4f})\n")

        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
        self._plot(y_test, y_pred, y_prob, results)
        return self

    def predict(self, df):
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
        print(f"  Numeric columns:     {num_cols}")
        print(f"  Categorical columns: {cat_cols}\n")
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers = []
        if num_cols: transformers.append(("num", num_pipe, num_cols))
        if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
        return ColumnTransformer(transformers)

    def _plot(self, y_test, y_pred, y_prob, results):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Churn Model — Winner: {self.best_name}", fontsize=13, fontweight="bold")

        best = max(results.values())
        colors = ["#2ecc71" if v == best else "#cccccc" for v in results.values()]
        bars = axes[0].barh(list(results.keys()), list(results.values()), color=colors)
        axes[0].set_xlim(0, 1.1)
        axes[0].set_title("Algorithm Comparison")
        for bar, v in zip(bars, results.values()):
            axes[0].text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center")

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["Retained", "Churned"],
            ax=axes[1], colorbar=False, cmap="Blues"
        )
        axes[1].set_title("Confusion Matrix")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[2].plot(fpr, tpr, color="#2ecc71", lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
        axes[2].plot([0, 1], [0, 1], "k--", lw=1)
        axes[2].set_xlabel("False Positive Rate")
        axes[2].set_ylabel("True Positive Rate")
        axes[2].set_title("ROC Curve")
        axes[2].legend()

        plt.tight_layout()
        plt.show()
