# ML Ops Agent

An ML platform that automatically finds the best algorithm for your business problem. You provide a cleaned dataset, and the agent tests multiple algorithms against each other, tunes each one, and picks the best performer.

---

## How it works

For each model, the agent runs **3 algorithms** (XGBoost, Random Forest, Logistic Regression). Each algorithm is tuned with **10 different hyperparameter combinations** tested via **5-fold cross-validation**. Whichever algorithm scores highest on your data wins.

```
Your dataset
    ↓
XGBoost       → 10 hyperparameter combos, 5-fold CV → best score
Random Forest → 10 hyperparameter combos, 5-fold CV → best score
Logistic Reg  → 10 hyperparameter combos, 5-fold CV → best score
    ↓
Winner selected automatically
    ↓
Predictions returned in plain business terms
```

---

## Data requirements

- Your dataset must be **cleaned** before passing it in (no nulls, correct column types)
- You must specify which column is the target (what you want to predict)
- **If your data is not compatible with the model you're using, it will raise a clear error and stop** — for example, passing a dataset with no text column to SentimentModel, or passing non-binary labels to ChurnModel

> Automatic data cleaning is planned for a future version.

---

## Models

### Churn Prediction — `churn/churn_model.py`

Predicts which customers are at risk of leaving.

**Compatible data:** Any tabular dataset with customer features and a binary churn column (0 = stayed, 1 = churned)

```python
from churn.churn_model import ChurnModel

model = ChurnModel(target_col="churned", drop_cols=["customer_id"])
model.fit(df)
predictions = model.predict(new_df)
# Returns: churn_probability, will_churn, risk_tier (Low / Medium / High)
```

---

### Sentiment Analysis — `sentiment/sentiment_model.py`

Classifies text (reviews, feedback, comments) as positive or negative.

**Compatible data:** Any dataset with a text column and a binary label column (0/1 or "positive"/"negative")

```python
from sentiment.sentiment_model import SentimentModel

model = SentimentModel(text_col="review", target_col="sentiment")
model.fit(df)
predictions = model.predict(new_df)
# Returns: text, label, confidence
```

---

### Fraud Detection — `fraud/fraud_model.py` *(coming soon)*

Identifies fraudulent transactions. Designed for datasets where fraud is <1% of records.

Planned algorithms: Isolation Forest + Random Forest

---

### Customer Segmentation — `segmentation/segmentation_model.py` *(coming soon)*

Groups customers into segments with no target label needed (unsupervised).

Planned algorithms: K-Means + DBSCAN

---

## Project structure

```
ml-ops-agent/
├── churn/
│   ├── churn_model.py       ✅ complete
│   └── churn_notebook.ipynb
├── sentiment/
│   ├── sentiment_model.py   ✅ complete
│   └── sentiment_notebook.ipynb
├── fraud/
│   └── fraud_model.py       coming soon
├── segmentation/
│   └── segmentation_model.py  coming soon
└── agent/
    └── router.py            coming soon — LLM-powered routing
```

---

## LLM Router *(coming soon)*

Instead of picking the model yourself, you'll describe your problem in plain English and the agent routes you automatically:

- *"Which customers are about to cancel?"* → ChurnModel
- *"Are these reviews positive or negative?"* → SentimentModel
- *"Flag suspicious transactions"* → FraudModel
- *"Group my customers into segments"* → SegmentationModel
