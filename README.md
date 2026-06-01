# ML Ops Agent

Describe your business problem in plain English and give it a dataset — the agent picks the right model and the best algorithm automatically.

---

## UI (recommended)

```
.\.venv\Scripts\streamlit.exe run app.py
```

Opens at `http://localhost:8501`. Three tabs:

- **Train** — describe your problem, drag and drop a CSV, configure columns, train. Downloads a `.pkl` of the trained model.
- **Predict** — upload a `.pkl` + new CSV, get predictions back as a downloadable CSV.
- **About** — model descriptions, data requirements, usage guide.

---

## How it works

The router maps your problem description to one of four models. The model then tests multiple algorithms on your data, tunes each one, and picks whichever performs best.

```python
from agent.router import route_and_get_class

result = route_and_get_class("Which customers are likely to cancel?")
ModelClass = result["model_class"]

model = ModelClass(target_col="churned", drop_cols=["customer_id"])
model.fit(df)
predictions = model.predict(new_df)

print(model.best_name)   # e.g. "XGBoost"
print(model.best_score)  # e.g. 0.8921
```

The router works out of the box with keyword matching. Set `ANTHROPIC_API_KEY` in your environment to use Claude for smarter routing.

To reuse a trained model later:

```python
import joblib
model = joblib.load("my_model.pkl")
predictions = model.predict(new_df)
```

---

## Models

### Churn Prediction
**File:** `churn/churn_model.py` | **Notebook:** `churn/churn_notebook.ipynb`

Predicts which customers are likely to leave.

**Algorithms:** XGBoost, Random Forest, Logistic Regression (10 configs each, 5-fold CV) — best ROC-AUC wins  
**Data:** Tabular dataset with customer features + binary churn column (0 = stayed, 1 = left)

```python
from churn.churn_model import ChurnModel

model = ChurnModel(target_col="churned", drop_cols=["customer_id"])
model.fit(df)
predictions = model.predict(new_df)
# Returns: churn_probability, will_churn, risk_tier (Low / Medium / High Risk)
```

---

### Sentiment Analysis
**File:** `sentiment/sentiment_model.py` | **Notebook:** `sentiment/sentiment_notebook.ipynb`

Classifies text (reviews, feedback, comments) as positive or negative.

**Algorithms:** Logistic Regression, Linear SVC, Naive Bayes, Random Forest, XGBoost (10 configs each, 5-fold CV) — best ROC-AUC wins  
**Data:** Dataset with a text column and a binary label column (0/1 or "positive"/"negative")

```python
from sentiment.sentiment_model import SentimentModel

model = SentimentModel(text_col="review", target_col="sentiment")
model.fit(df)
predictions = model.predict(new_df["review"])
# Returns: text, label (positive / negative), confidence score
```

---

### Fraud Detection
**File:** `fraud/fraud_model.py`

Detects fraudulent transactions. Built for datasets where fraud is a small fraction of records — all algorithms are configured to handle class imbalance. Uses Precision-Recall AUC as the scoring metric, which is more meaningful than ROC-AUC on imbalanced data.

**Algorithms:** XGBoost, Random Forest, Logistic Regression (all with imbalance handling, 10 configs each, 5-fold CV)  
**Data:** Tabular transaction dataset with binary fraud column (0 = legitimate, 1 = fraud)

```python
from fraud.fraud_model import FraudModel

model = FraudModel(target_col="is_fraud", drop_cols=["transaction_id"])
model.fit(df)
predictions = model.predict(new_df)
# Returns: fraud_probability, is_fraud
```

---

### Customer Segmentation
**File:** `segmentation/segmentation_model.py`

Groups customers into natural segments. No target label needed — it finds the groupings automatically.

**Algorithms:** K-Means (k=2–8) and DBSCAN (multiple settings) — best silhouette score wins  
**Data:** Tabular dataset with numeric features. No target column required.

```python
from segmentation.segmentation_model import SegmentationModel

model = SegmentationModel(drop_cols=["customer_id"])
model.fit(df)
segments = model.predict(new_df)
# Returns: segment number per row

profiles = model.segment_profiles()
# Returns: mean feature values per segment
```

---

## Data requirements

- Models handle missing values via imputation, but clean data produces better results
- Specify `target_col` for all models except Segmentation
- If your data isn't compatible, the model will stop and tell you exactly why

---

## Project structure

```
ml-ops-agent/
├── app.py                     # Streamlit UI
├── agent/
│   └── router.py              # routes plain-English problems to the right model
├── churn/
│   ├── churn_model.py
│   └── churn_notebook.ipynb
├── sentiment/
│   ├── sentiment_model.py
│   └── sentiment_notebook.ipynb
├── fraud/
│   └── fraud_model.py
├── segmentation/
│   └── segmentation_model.py
└── run_smoke_tests.py         # end-to-end tests for all models + router
```
