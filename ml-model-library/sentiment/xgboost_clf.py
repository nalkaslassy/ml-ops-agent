from xgboost import XGBClassifier

NAME = "XGBoost"

def get_estimator():
    return XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)

def get_params():
    return {
        "classifier__n_estimators":  [100, 200, 300],
        "classifier__max_depth":     [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
    }
