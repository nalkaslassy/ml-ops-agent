# ML Ops Agent

Give it a cleaned dataset, tell it what you want to predict, and it figures out the best ML algorithm for your data automatically.

---

## How it works

The agent runs 3 algorithms on your data — XGBoost, Random Forest, and Logistic Regression. Each one is tested with 10 different settings using 5-fold cross-validation. The one that scores highest on your specific dataset wins and is used for predictions.

You don't pick the algorithm. The agent does.

---

## Important: data requirements

- Your dataset must be **cleaned** before using it (no missing values, correct column types)
- You need to tell it which column you want to predict
- If your data isn't compatible with the model you're using, it will stop and tell you exactly why

> Automatic data cleaning is planned for a future version.

---

## What's available

### Churn Prediction ✅
**File:** `churn/churn_model.py`

Predicts which customers are likely to leave. Give it a dataset with customer features and a column marking who churned (0 = stayed, 1 = left).

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

Classifies text as positive or negative. Works on reviews, feedback, comments — anything with a text column and a label.

```python
from sentiment.sentiment_model import SentimentModel

model = SentimentModel(text_col="review", target_col="sentiment")
model.fit(df)
predictions = model.predict(new_df)
# Returns: text, label (Positive / Negative), confidence score
```

Labels can be 0/1 or strings like "positive"/"negative" — either works.

---

### Fraud Detection 🔜
**File:** `fraud/fraud_model.py`

Coming soon. Built for transaction datasets where fraud is a small fraction of records.

---

### Customer Segmentation 🔜
**File:** `segmentation/segmentation_model.py`

Coming soon. Groups customers into segments — no target label needed.

---

## Project structure

```
ml-ops-agent/
├── churn/
│   ├── churn_model.py        ✅ ready
│   └── churn_notebook.ipynb
├── sentiment/
│   ├── sentiment_model.py    ✅ ready
│   └── sentiment_notebook.ipynb
├── fraud/
│   └── fraud_model.py        coming soon
├── segmentation/
│   └── segmentation_model.py coming soon
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
