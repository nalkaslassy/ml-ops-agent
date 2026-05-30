import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from xgboost import XGBClassifier


ALGORITHMS = [
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "classifier__C":      [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs", "saga"],
        }
    },
    {
        "name": "Random Forest",
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "classifier__n_estimators":     [100, 200, 300],
            "classifier__max_depth":        [None, 10, 20],
            "classifier__min_samples_split":[2, 5, 10],
        }
    },
    {
        "name": "XGBoost",
        "model": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
        "params": {
            "classifier__n_estimators":  [100, 200, 300],
            "classifier__max_depth":     [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
        }
    },
]


class SentimentModel:
    """
    Classifies text as positive or negative (or any labels you provide).
    Works with any dataset that has a text column and a label column.
    Automatically cleans text, builds TF-IDF features, tests all 3 algorithms,
    tunes each one, and picks the best.
    """

    def __init__(self, text_col="text", target_col="sentiment", label_names=None):
        self.text_col    = text_col
        self.target_col  = target_col
        self.label_names = label_names or ["Negative", "Positive"]
        self.vectorizer  = None
        self.encoder     = None
        self.best_model  = None
        self.best_name   = None

    def _validate(self, df):
        if len(df) < 200:
            raise ValueError(
                f"Dataset has only {len(df)} rows — at least 200 are required to train reliably."
            )
        if self.text_col not in df.columns:
            raise ValueError(
                f"Text column '{self.text_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        if not pd.api.types.is_string_dtype(df[self.text_col]) and not pd.api.types.is_object_dtype(df[self.text_col]):
            raise ValueError(
                f"Text column '{self.text_col}' must contain strings, "
                f"but has dtype '{df[self.text_col].dtype}'."
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

    def fit(self, df):
        self._validate(df)
        texts = df[self.text_col].astype(str).apply(self._clean_text)
        y     = df[self.target_col]

        # Encode string labels to integers if needed
        if y.dtype == object:
            self.encoder = LabelEncoder()
            y = self.encoder.fit_transform(y)
            self.label_names = list(self.encoder.classes_)

        print(f"Training on {len(texts)} rows | Labels: {self.label_names}\n")

        # Convert text to TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        X = self.vectorizer.fit_transform(texts)

        print(f"  TF-IDF features: {X.shape[1]}\n")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        best_score = -1
        results = {}

        for algo in ALGORITHMS:
            pipeline = Pipeline([("classifier", algo["model"])])
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
        print(classification_report(y_test, y_pred, target_names=self.label_names))
        self._plot(y_test, y_pred, y_prob, results)
        return self

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        cleaned = [self._clean_text(t) for t in texts]
        X       = self.vectorizer.transform(cleaned)
        probs   = self.best_model.predict_proba(X)[:, 1]
        preds   = self.best_model.predict(X)
        labels  = [self.label_names[p] for p in preds]

        return pd.DataFrame({
            "text":       texts,
            "label":      labels,
            "confidence": np.round(np.maximum(probs, 1 - probs), 4),
        })

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _plot(self, y_test, y_pred, y_prob, results):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Sentiment Model — Winner: {self.best_name}", fontsize=13, fontweight="bold")

        best = max(results.values())
        colors = ["#e74c3c" if v == best else "#cccccc" for v in results.values()]
        bars = axes[0].barh(list(results.keys()), list(results.values()), color=colors)
        axes[0].set_xlim(0, 1.1)
        axes[0].set_title("Algorithm Comparison")
        for bar, v in zip(bars, results.values()):
            axes[0].text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center")

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=self.label_names,
            ax=axes[1], colorbar=False, cmap="Reds"
        )
        axes[1].set_title("Confusion Matrix")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[2].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
        axes[2].plot([0, 1], [0, 1], "k--", lw=1)
        axes[2].set_xlabel("False Positive Rate")
        axes[2].set_ylabel("True Positive Rate")
        axes[2].set_title("ROC Curve")
        axes[2].legend()

        plt.tight_layout()
        plt.show()
