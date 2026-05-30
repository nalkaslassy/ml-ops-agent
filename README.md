# ML Ops Agent

An extensible ML platform that automatically selects the best algorithm for your business problem. You provide a cleaned dataset, specify what you want to predict, and the agent handles everything else — competing algorithms, hyperparameter tuning, and model selection.

---

## How It Works

```
You describe the task (e.g. churn prediction)
        ↓
Provide a cleaned, compatible dataset
        ↓
The agent runs all algorithms in the registry for that task:
    ├── XGBoost       → 10 random hyperparameter combos, 5-fold CV
    ├── Random Forest → 10 random hyperparameter combos, 5-fold CV
    └── Logistic Regression → 10 random hyperparameter combos, 5-fold CV
        ↓
Best algorithm + best hyperparameters are selected automatically
        ↓
Returns predictions in plain business terms (e.g. risk tier, probability, label)
```

**The agent uses a plugin architecture** — every task has its own folder with a registry of competing algorithms. Adding a new algorithm to any task is as simple as dropping a new file in its `algorithms/` folder.

**Coming soon: LLM-powered router** — instead of picking the model yourself, you'll describe your business problem in plain English (e.g. *"which customers are likely to cancel?"*) and the agent will automatically route you to the right model.

---

## Data Requirements

> **Your data must be cleaned before passing it to the agent.**

The agent does not handle raw or messy data at this stage. Before using any model:

- Remove or impute missing values
- Ensure column types are correct (numbers as numeric, text as string)
- Ensure the target column contains the labels you want to predict

**If the dataset is not compatible with the requested task, the model will raise a validation error and not run.** For example, passing a dataset with no text column to the SentimentModel, or passing a dataset without a binary target column to the ChurnModel, will fail with a clear explanation.

> **Planned:** Automatic data cleaning and compatibility checking is on the roadmap — the agent will eventually handle dirty data and guide you through fixes automatically.

---

## Models

### Done

#### ChurnModel — Customer Churn Prediction
Predicts which customers are at risk of leaving.

- **Input:** Tabular dataset with numeric and/or categorical customer features + a binary churn target column
- **Algorithms:** XGBoost, Random Forest, Logistic Regression
- **Output:** `churn_probability`, `will_churn`, `risk_tier` (Low / Medium / High)
- **Compatible data:** Any dataset with a binary churn/retention label (e.g. telecom, SaaS, e-commerce)

```python
from ml_model_library.models.churn.churn_model import ChurnModel

model = ChurnModel(target_col="churned", drop_cols=["customer_id"])
model.fit(df)
predictions = model.predict(new_df)
```

---

#### SentimentModel — Sentiment Analysis
Classifies text (reviews, feedback, comments) as positive or negative.

- **Input:** Dataset with a text column and a sentiment label column
- **Algorithms:** XGBoost, Random Forest, Logistic Regression (over TF-IDF features)
- **Output:** `label`, `confidence`

```python
from sentiment.sentiment_model import SentimentModel

model = SentimentModel(text_col="review", target_col="sentiment")
model.fit(df)
predictions = model.predict(new_df)
```

---

### Planned

#### FraudModel — Fraud Detection
Identifies fraudulent transactions in a dataset with extreme class imbalance.

- **Algorithms planned:** Isolation Forest (unsupervised) + Random Forest (supervised)
- **Metric:** Precision-Recall AUC (more appropriate than ROC-AUC for heavily imbalanced data)
- **Compatible data:** Transaction datasets where fraud is <1% of records

#### SegmentationModel — Customer Segmentation
Discovers natural customer groups without a target label (unsupervised).

- **Algorithms planned:** K-Means, DBSCAN
- **Selection:** Silhouette score used to pick optimal number of clusters
- **Output:** Cluster labels + segment profiles for marketing or targeting

#### LLM Router
Routes plain-English business problems to the correct model automatically.

- **Example:** *"Which of my customers are about to cancel?"* → `ChurnModel`
- **Example:** *"Are my product reviews mostly positive or negative?"* → `SentimentModel`
- **Built with:** Claude API
- **Status:** Planned — will be built once core models are complete

---

## Project Structure

```
ml-ops-agent/
├── ml-model-library/
│   ├── models/
│   │   ├── churn/                  ✅ Complete
│   │   │   ├── algorithms/
│   │   │   │   ├── xgboost_clf.py
│   │   │   │   ├── random_forest_clf.py
│   │   │   │   └── logistic_clf.py
│   │   │   └── churn_model.py
│   │   ├── sentiment/              ✅ Complete
│   │   │   ├── algorithms/
│   │   │   └── sentiment_model.py
│   │   ├── fraud/                  📋 Planned
│   │   └── segmentation/           📋 Planned
│   ├── notebooks/                  # Jupyter notebooks for each model
│   └── data/sample_datasets/       # Drop your CSV files here
└── agent/
    └── router.py                   📋 Planned — LLM-powered routing
```

---

## Adding a New Algorithm

Drop a file in the relevant `algorithms/` folder:

```python
# ml-model-library/models/churn/algorithms/lightgbm_clf.py

ALGORITHM_NAME = "LightGBM"

def get_estimator():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=42)

def get_param_grid():
    return {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth":     [3, 5, 7],
    }
```

Then register it in `algorithms/__init__.py` and it will automatically compete alongside the existing algorithms.

---

## Tech Stack

- `scikit-learn` — pipelines, preprocessing, cross-validation
- `xgboost` — gradient boosting
- `pandas` / `numpy` — data handling
- `matplotlib` — visualizations (ROC curves, confusion matrices, algorithm comparison)
- `jupyter` — interactive notebooks for experimentation
