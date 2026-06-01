# ML Ops Agent

Give it a dataset and describe what you want to do — it picks the right model, tests multiple algorithms, and returns the best one for your data automatically.

---

## Getting started

```
.\.venv\Scripts\streamlit.exe run app.py
```

Opens at `http://localhost:8501`.

---

## How it works

1. **Describe your problem** — type something like *"which customers are likely to cancel?"* or *"detect fraudulent transactions"*
2. **Upload your CSV** — drag and drop your dataset
3. **Configure** — pick which column to predict and which columns to ignore (IDs etc.)
4. **Train** — the agent tests every algorithm across many settings, picks the one that scores best on your data, and shows you the results
5. **Download the model** — save the `.pkl` file to reuse for predictions later

---

## Predict on new data

Upload your saved `.pkl` and a new CSV in the **Predict** tab — get predictions back as a downloadable CSV. No retraining needed.

You can also use a saved model directly in Python:

```python
import joblib
model = joblib.load("my_model.pkl")
predictions = model.predict(new_df)
```

---

## What gets returned

After training you can see:
- Which algorithm won and what score it got
- A chart comparing all algorithms that were tested
- Detailed metrics (confusion matrix, ROC/PR curve)

Predictions include a probability score per row so you know how confident the model is:

| Model | Prediction output |
|---|---|
| Churn | churn_probability, will_churn, risk_tier (Low / Medium / High) |
| Sentiment | label (positive / negative), confidence score |
| Fraud | fraud_probability, is_fraud |
| Segmentation | segment number |

---

## Supported models

**Churn Prediction** — predicts which customers are likely to leave  
Data: customer features + binary churn column (0 = stayed, 1 = left)

**Sentiment Analysis** — classifies text as positive or negative  
Data: text column + binary label column (0/1 or "positive"/"negative")

**Fraud Detection** — detects fraudulent transactions, built for imbalanced data where fraud is rare  
Data: transaction features + binary fraud column (0 = legitimate, 1 = fraud)

**Customer Segmentation** — groups customers into natural segments, no label needed  
Data: numeric feature columns only

---

## Algorithms tested per model

| Model | Algorithms | Scored by |
|---|---|---|
| Churn | XGBoost, Random Forest, Logistic Regression | ROC-AUC |
| Sentiment | Logistic Regression, Linear SVC, Naive Bayes, Random Forest, XGBoost | ROC-AUC |
| Fraud | XGBoost, Random Forest, Logistic Regression (all imbalance-aware) | Precision-Recall AUC |
| Segmentation | K-Means (k=2–8), DBSCAN (multiple settings) | Silhouette score |

Each algorithm is tuned across multiple hyperparameter configurations using cross-validation. The one with the best score wins.

---

## Data requirements

- Models handle missing values automatically — clean data gives better results
- All models except Segmentation need a `target_col` (the column to predict)
- If your data isn't compatible the model will tell you exactly what's wrong

---

## Project structure

```
ml-ops-agent/
├── app.py                        # Streamlit UI
├── agent/
│   └── router.py                 # maps plain-English problems to the right model
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
└── run_smoke_tests.py            # end-to-end tests for all models and the router
```
