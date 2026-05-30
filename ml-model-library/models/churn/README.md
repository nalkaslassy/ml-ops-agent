# Customer Churn Prediction

Auto-adaptive churn prediction — works with telecom, SaaS, e-commerce, banking, or any domain.

---

## Files

```
churn/
├── algorithms/
│   ├── __init__.py          — Algorithm registry (add new algorithms here)
│   ├── xgboost_clf.py       — XGBoost
│   ├── random_forest_clf.py — Random Forest
│   └── logistic_clf.py      — Logistic Regression
├── churn_model.py           — ChurnModel orchestrator
├── __init__.py
└── README.md
```

---

## Quickstart

```python
import pandas as pd
from models.churn import ChurnModel

df = pd.read_csv('your_churn_data.csv')

model = ChurnModel(target_col='churned', drop_cols=['customer_id'])
model.fit(df)

predictions = model.predict(new_customers)
print(predictions)
# Returns: churn_probability | will_churn | risk_tier
```

---

## Adding a New Algorithm

1. Create a new file in `algorithms/`, e.g. `lightgbm_clf.py`
2. Define `ALGORITHM_NAME`, `get_estimator()`, and `get_param_grid()`
3. Register it in `algorithms/__init__.py`

That's it — ChurnModel will automatically include it in the next run.

```python
# algorithms/lightgbm_clf.py
from lightgbm import LGBMClassifier

ALGORITHM_NAME = "LightGBM"

def get_estimator():
    return LGBMClassifier(random_state=42)

def get_param_grid():
    return {
        "classifier__n_estimators":  [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__max_depth":     [3, 5, 7],
    }
```

---

## Public Datasets to Test With

| Dataset | Target Col | Link |
|---|---|---|
| IBM Telco Churn | `Churn` | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| E-Commerce Churn | `Churn` | https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction |
| SaaS Churn | `churned` | https://www.kaggle.com/datasets/gsagar12/dspp1 |
