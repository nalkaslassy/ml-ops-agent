import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


ALGORITHMS = [
    {
        "name": "XGBoost",
        "model": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
        "params": {
            "classifier__n_estimators":    [100, 200, 300, 500],
            "classifier__max_depth":       [3, 4, 5, 6, 8],
            "classifier__learning_rate":   [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample":       [0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree":[0.7, 0.8, 0.9, 1.0],
            "classifier__min_child_weight":[1, 3, 5],
        }
    },
    {
        "name": "Random Forest",
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
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
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "classifier__C":       [0.01, 0.1, 1, 10, 100],
            "classifier__solver":  ["lbfgs", "saga"],
            "classifier__penalty": ["l2"],
        }
    },
]


class ChurnModel:
    """
    Predicts customer churn. Works with any dataset.
    Set target_col to your churn label column and drop_cols to any ID columns.
    Automatically detects features, tunes each algorithm, and picks the best one.
    """

    def __init__(self, target_col="churned", drop_cols=None, test_size=0.2,
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
        self.results_     = {}
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
            sample = sorted(str(v) for v in unique_vals)[:5]
            suffix = f" … ({len(unique_vals)} unique values total)" if len(unique_vals) > 5 else ""
            raise ValueError(
                f"Target column '{self.target_col}' must be binary (0 = retained, 1 = churned), "
                f"but found: {sample}{suffix}"
            )
        if len(unique_vals) < 2:
            raise ValueError(
                f"Target column '{self.target_col}' contains only one class ({list(unique_vals)[0]}). "
                f"Both 0 (retained) and 1 (churned) must be present to train."
            )
        drop = [self.target_col] + [c for c in self.drop_cols if c in df.columns]
        feature_cols = [c for c in df.columns if c not in drop]
        if len(feature_cols) < 2:
            raise ValueError(
                f"At least 2 feature columns required after dropping target and drop_cols, "
                f"but only {len(feature_cols)} remain: {feature_cols}"
            )

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

        churn_rate = y.mean()
        print(f"Training on {len(X):,} rows | {len(X.columns)} features | Churn rate: {churn_rate:.1%}\n")
        if churn_rate < 0.05 or churn_rate > 0.95:
            print("  Warning: Class imbalance detected — consider SMOTE for better minority recall.\n")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        preprocessor = self._build_preprocessor(X_train)
        best_score = -1
        results = {}

        print(f"Testing {len(ALGORITHMS)} algorithms ({self.n_iter} configs x {self.cv}-fold CV each)...\n")

        for algo in ALGORITHMS:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier",   algo["model"]),
            ])
            search = RandomizedSearchCV(
                pipeline, algo["params"],
                n_iter=self.n_iter, cv=self.cv, scoring="roc_auc",
                random_state=self.random_state, n_jobs=-1
            )
            search.fit(X_train, y_train)
            score = search.best_score_
            results[algo["name"]] = score
            marker = "  <- best so far" if score > best_score else ""
            print(f"  {algo['name']:<25} AUC = {score:.4f}{marker}")

            if score > best_score:
                best_score        = score
                self.best_model   = search.best_estimator_
                self.best_name    = algo["name"]
                self._best_params = search.best_params_

        self.best_score = best_score
        self.results_   = results
        print(f"\nWinner: {self.best_name} (AUC: {best_score:.4f})")
        print(f"Params: {self._best_params}\n")

        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        self.test_score = roc_auc_score(y_test, y_prob)
        print(f"Test AUC: {self.test_score:.4f}\n")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
        self.is_fitted = True
        return self

    def predict(self, df):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X     = df.drop(columns=[c for c in [self.target_col] + self.drop_cols if c in df.columns])
        probs = self.best_model.predict_proba(X)[:, 1]
        preds = self.best_model.predict(X)
        tiers = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0], labels=["Low Risk", "Medium Risk", "High Risk"])
        return pd.DataFrame({
            "churn_probability": np.round(probs, 4),
            "will_churn":        preds,
            "risk_tier":         tiers,
        }, index=df.index)
