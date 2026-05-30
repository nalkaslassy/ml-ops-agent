from sklearn.linear_model import LogisticRegression

NAME = "Logistic Regression"

def get_estimator():
    return LogisticRegression(random_state=42, max_iter=1000)

def get_params():
    return {
        "classifier__C":      [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["lbfgs", "saga"],
    }
