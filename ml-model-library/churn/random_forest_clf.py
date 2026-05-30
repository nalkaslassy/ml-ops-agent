from sklearn.ensemble import RandomForestClassifier

NAME = "Random Forest"

def get_estimator():
    return RandomForestClassifier(random_state=42, n_jobs=-1)

def get_params():
    return {
        "classifier__n_estimators":     [100, 200, 300],
        "classifier__max_depth":        [None, 5, 10, 20],
        "classifier__min_samples_split":[2, 5, 10],
    }
