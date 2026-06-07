import re
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


ALGORITHMS = [
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "classifier__C":      [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs", "saga"],
            "classifier__penalty":["l2"],
        }
    },
    {
        "name": "Linear SVC",
        "model": CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000)),
        "params": {
            "classifier__estimator__C": [0.01, 0.1, 1, 10, 100],
        }
    },
    {
        "name": "Naive Bayes",
        "model": MultinomialNB(),
        "params": {
            "classifier__alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
        }
    },
    {
        "name": "Random Forest",
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "classifier__n_estimators":      [100, 200, 300],
            "classifier__max_depth":         [None, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf":  [1, 2, 4],
            "classifier__max_features":      ["sqrt", "log2"],
        }
    },
    {
        "name": "XGBoost",
        "model": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
        "params": {
            "classifier__n_estimators":    [100, 200, 300],
            "classifier__max_depth":       [3, 5, 7],
            "classifier__learning_rate":   [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample":       [0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree":[0.7, 0.8, 0.9, 1.0],
        }
    },
]


class SentimentModel:
    """
    Classifies text as positive or negative (or any two labels you provide).
    Set text_col to your text column and target_col to your label column.
    Automatically cleans text, builds TF-IDF features, tests all algorithms,
    tunes each one, and picks the best.
    """

    def __init__(self, text_col="text", target_col="sentiment", label_names=None,
                 test_size=0.2, random_state=42, n_iter=10, cv=5):
        self.text_col     = text_col
        self.target_col   = target_col
        self.label_names  = label_names or ["Negative", "Positive"]
        self.test_size    = test_size
        self.random_state = random_state
        self.n_iter       = n_iter
        self.cv           = cv
        self.vectorizer   = None
        self.encoder      = None
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
        if self.text_col not in df.columns:
            raise ValueError(
                f"Text column '{self.text_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        if not (pd.api.types.is_string_dtype(df[self.text_col]) or
                pd.api.types.is_object_dtype(df[self.text_col])):
            raise ValueError(
                f"Text column '{self.text_col}' must contain strings, "
                f"but has dtype '{df[self.text_col].dtype}'."
            )
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        n_unique = df[self.target_col].nunique()
        if n_unique != 2:
            raise ValueError(
                f"Target column '{self.target_col}' must have exactly 2 unique values "
                f"(e.g. 0/1 or 'positive'/'negative'), but found {n_unique}: "
                f"{sorted(df[self.target_col].dropna().unique())}"
            )

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(self, df):
        self._validate(df)
        texts = df[self.text_col].apply(self._clean_text)
        y     = df[self.target_col]

        if not pd.api.types.is_numeric_dtype(y):
            self.encoder = LabelEncoder()
            y = self.encoder.fit_transform(y)
            self.label_names = list(self.encoder.classes_)

        print(f"Training on {len(texts):,} rows | Labels: {self.label_names}\n")

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        X = self.vectorizer.fit_transform(texts)
        print(f"  TF-IDF features: {X.shape[1]}\n")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        best_score = -1
        results = {}

        print(f"Testing {len(ALGORITHMS)} algorithms ({self.n_iter} configs x {self.cv}-fold CV each)...\n")

        for algo in ALGORITHMS:
            pipeline = Pipeline([("classifier", algo["model"])])
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
        print(classification_report(y_test, y_pred, target_names=self.label_names))
        self.is_fitted = True
        return self

    def predict(self, texts):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
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
