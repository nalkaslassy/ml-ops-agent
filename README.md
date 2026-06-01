# ML Ops Agent

Give it a cleaned dataset, tell it what you want to predict, and it figures out the best ML algorithm for your data automatically.

---

## How it works

The agent runs multiple algorithms on your data, tests each one with different settings, and picks whichever scores best on your specific dataset. You don't choose the algorithm — the agent does.

---

## Important: data requirements

- Your dataset must be **cleaned** before using it (no missing values, correct column types)
- You need to tell it which column you want to predict (except Segmentation — that one needs no target)
- If your data isn't compatible with the model you're using, it will stop and tell you exactly why

> Automatic data cleaning is planned for a future version.

---

## Models

### Churn Prediction ✅
**File:** `churn/churn_model.py`

Predicts which customers are likely to leave.

**Algorithms tested:** XGBoost, Random Forest, Logistic Regression (10 settings each, 5-fold CV)
**Compatible data:** Tabular dataset with customer features + a binary churn column (0 = stayed, 1 = left)

```python
from churn.churn_model import ChurnModel

model = ChurnModel(target_col="churned", drop_cols=["customer_id"])
model.fit(df)
predictions = model.predict(new_df)
# Returns: churn_probability, will_churn, risk_tier (Low / Medium / High Risk)
```

---

### Sentiment Analysis ✅
**File:** `sentiment/sentiment_model.py`

Classifies text (reviews, feedback, comments) as positive or negative.

**Algorithms tested:** XGBoost, Random Forest, Logistic Regression (10 settings each, 5-fold CV)
**Compatible data:** Dataset with a text column and a binary label column (0/1 or "positive"/"negative")

```python
from sentiment.sentiment_model import SentimentModel

model = SentimentModel(text_col="review", target_col="sentiment")
model.fit(df)
predictions = model.predict(new_df)
# Returns: text, label (Positive / Negative), confidence score
```

---

### Fraud Detection ✅
**File:** `fraud/fraud_model.py`

Identifies fraudulent transactions. Built specifically for datasets where fraud is a tiny fraction of records — all 3 algorithms are configured to handle that imbalance automatically. Uses Precision-Recall AUC as the scoring metric, which is more meaningful than ROC-AUC for imbalanced data.

**Algorithms tested:** XGBoost, Random Forest, Logistic Regression (all with imbalance handling, 10 settings each, 5-fold CV)
**Compatible data:** Tabular transaction dataset with a binary fraud column (0 = legitimate, 1 = fraud)

```python
from fraud.fraud_model import FraudModel

model = FraudModel(target_col="is_fraud", drop_cols=["transaction_id"])
model.fit(df)
predictions = model.predict(new_df)
# Returns: fraud_probability, is_fraud
```

---

### Customer Segmentation ✅
**File:** `segmentation/segmentation_model.py`

Groups customers into segments. No target label needed — it finds the natural groupings in your data automatically.

**Algorithms tested:** K-Means (k=2 through 8) and DBSCAN (multiple settings) — best silhouette score wins
**Compatible data:** Tabular dataset with numeric customer features. No target column required.

```python
from segmentation.segmentation_model import SegmentationModel

model = SegmentationModel(drop_cols=["customer_id"])
model.fit(df)
segments = model.predict(new_df)
# Returns: segment number per row

profiles = model.segment_profiles()
# Returns: mean feature values per segment (useful for understanding each group)
```

---

## Project structure

```
ml-ops-agent/
├── churn/
│   ├── churn_model.py        ✅
│   └── churn_notebook.ipynb
├── sentiment/
│   ├── sentiment_model.py    ✅
│   └── sentiment_notebook.ipynb
├── fraud/
│   └── fraud_model.py        ✅
├── segmentation/
│   └── segmentation_model.py ✅
└── agent/
    └── router.py             coming soon
```

---

## Coming soon: plain-English routing

Instead of importing the model yourself, you'll be able to describe your problem and the agent will pick the right model automatically.

- *"Which customers are about to cancel?"* → Churn
- *"Are these reviews positive or negative?"* → Sentiment
- *"Find suspicious transactions"* → Fraud
- *"Group my customers"* → Segmentation
