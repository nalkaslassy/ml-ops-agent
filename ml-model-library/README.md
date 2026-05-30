# ML Model Library

Two plug-and-play ML models for business use cases.

```
ml-model-library/
├── churn/
│   ├── xgboost_clf.py       — XGBoost algorithm
│   ├── random_forest_clf.py — Random Forest algorithm
│   ├── logistic_clf.py      — Logistic Regression algorithm
│   ├── churn_model.py       — ChurnModel class
│   └── churn_notebook.ipynb — Run this
├── sentiment/
│   ├── xgboost_clf.py       — XGBoost algorithm
│   ├── random_forest_clf.py — Random Forest algorithm
│   ├── logistic_clf.py      — Logistic Regression algorithm
│   ├── sentiment_model.py   — SentimentModel class (TODO)
│   └── sentiment_notebook.ipynb — Run this (TODO)
└── agent/                   — Coming later
```

## Quickstart

```python
# Churn
from churn.churn_model import ChurnModel
model = ChurnModel(target_col="churned", drop_cols=["customer_id"])
model.fit(df)
predictions = model.predict(new_df)

# Sentiment (coming soon)
from sentiment.sentiment_model import SentimentModel
model = SentimentModel(text_col="review", target_col="sentiment")
model.fit(df)
predictions = model.predict(["Great product!", "Terrible experience."])
```
