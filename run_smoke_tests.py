import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# Use a non-interactive backend so model plots do not block test execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.show = lambda *args, **kwargs: None

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.router import route_and_get_class


SEED = 42
rng = np.random.default_rng(SEED)


def make_churn_data(n=300):
    tenure = rng.integers(1, 72, n)
    monthly_spend = rng.normal(70, 20, n).clip(15, 250)
    support_tickets = rng.poisson(2, n)
    contract = rng.choice(["monthly", "annual"], size=n, p=[0.7, 0.3])

    logits = (
        -3.0
        + (monthly_spend - 70) * 0.03
        + support_tickets * 0.35
        + (contract == "monthly") * 1.1
        - tenure * 0.015
    )
    probs = 1 / (1 + np.exp(-logits))
    churned = (rng.random(n) < probs).astype(int)

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "tenure_months": tenure,
            "monthly_spend": monthly_spend.round(2),
            "support_tickets": support_tickets,
            "contract_type": contract,
            "churned": churned,
        }
    )


def make_fraud_data(n=400):
    amount = rng.lognormal(mean=3.7, sigma=0.8, size=n)
    hour = rng.integers(0, 24, n)
    is_international = rng.choice([0, 1], size=n, p=[0.92, 0.08])
    device_trust = rng.normal(0.75, 0.15, n).clip(0.0, 1.0)

    logits = (
        -5.2
        + (amount > 200).astype(int) * 1.1
        + ((hour <= 4) | (hour >= 23)).astype(int) * 0.9
        + is_international * 1.4
        + (device_trust < 0.4).astype(int) * 1.2
    )
    probs = 1 / (1 + np.exp(-logits))
    is_fraud = (rng.random(n) < probs).astype(int)

    # Ensure both classes exist for stable stratification/scoring.
    if is_fraud.sum() == 0:
        is_fraud[rng.choice(n, size=5, replace=False)] = 1

    return pd.DataFrame(
        {
            "transaction_id": np.arange(10_000, 10_000 + n),
            "amount": amount.round(2),
            "hour_of_day": hour,
            "is_international": is_international,
            "device_trust": device_trust.round(3),
            "is_fraud": is_fraud,
        }
    )


def make_sentiment_data(n=300):
    positive_templates = [
        "Great product and excellent quality",
        "Fast shipping and very happy with this purchase",
        "Customer support was helpful and friendly",
        "Love it, works perfectly",
        "Fantastic value and easy to use",
    ]
    negative_templates = [
        "Terrible experience and poor quality",
        "Late delivery and very disappointed",
        "Support was unhelpful and rude",
        "Hate it, stopped working quickly",
        "Waste of money and frustrating",
    ]

    labels = rng.choice([0, 1], size=n, p=[0.48, 0.52])
    texts = []
    for lbl in labels:
        if lbl == 1:
            txt = rng.choice(positive_templates)
            txt += f" rating {rng.integers(4, 6)} stars"
        else:
            txt = rng.choice(negative_templates)
            txt += f" rating {rng.integers(1, 3)} stars"
        texts.append(txt)

    sentiments = np.where(labels == 1, "positive", "negative")
    return pd.DataFrame({"review": texts, "sentiment": sentiments})


def make_segmentation_data(n=240):
    c1 = rng.normal(loc=[25, 30, 20], scale=[4, 6, 5], size=(n // 3, 3))
    c2 = rng.normal(loc=[45, 65, 55], scale=[5, 8, 7], size=(n // 3, 3))
    c3 = rng.normal(loc=[60, 20, 80], scale=[6, 5, 6], size=(n - 2 * (n // 3), 3))
    arr = np.vstack([c1, c2, c3])

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "annual_spend_k": arr[:, 0].round(2),
            "engagement_score": arr[:, 1].round(2),
            "lifetime_value_k": arr[:, 2].round(2),
        }
    )


def run_case(name, prompt, expected_model, df, init_kwargs, predict_input):
    print(f"\n=== {name} ===")
    routed = route_and_get_class(prompt, prefer_llm=False)
    actual_model = routed["model"]
    print(f"Router picked: {actual_model}")

    assert actual_model == expected_model, (
        f"Routing mismatch for {name}: expected {expected_model}, got {actual_model}"
    )

    model_class = routed["model_class"]
    model = model_class(**init_kwargs)
    model.fit(df)
    preds = model.predict(predict_input)

    assert len(preds) == len(predict_input), f"Unexpected prediction length for {name}"
    assert len(preds.columns) > 0, f"No prediction columns for {name}"

    print(f"Prediction shape: {preds.shape}")
    print(f"Prediction columns: {list(preds.columns)}")
    return True


def main():
    churn_df = make_churn_data()
    fraud_df = make_fraud_data()
    sentiment_df = make_sentiment_data()
    segment_df = make_segmentation_data()

    run_case(
        name="Churn",
        prompt="Which customers are likely to cancel their subscription soon?",
        expected_model="ChurnModel",
        df=churn_df,
        init_kwargs={"target_col": "churned", "drop_cols": ["customer_id"], "n_iter": 1, "cv": 2},
        predict_input=churn_df.head(20),
    )

    run_case(
        name="Fraud",
        prompt="Detect suspicious and potentially fraudulent transactions.",
        expected_model="FraudModel",
        df=fraud_df,
        init_kwargs={"target_col": "is_fraud", "drop_cols": ["transaction_id"], "n_iter": 1, "cv": 2},
        predict_input=fraud_df.head(20),
    )

    run_case(
        name="Sentiment",
        prompt="Classify customer review text as positive or negative.",
        expected_model="SentimentModel",
        df=sentiment_df,
        init_kwargs={"text_col": "review", "target_col": "sentiment", "n_iter": 1, "cv": 2},
        predict_input=sentiment_df["review"].head(20),
    )

    run_case(
        name="Segmentation",
        prompt="Group customers into natural segments for campaigns.",
        expected_model="SegmentationModel",
        df=segment_df,
        init_kwargs={"drop_cols": ["customer_id"]},
        predict_input=segment_df.head(20),
    )

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
