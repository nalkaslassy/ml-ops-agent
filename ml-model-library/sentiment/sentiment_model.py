"""
sentiment_model.py — TODO: implement SentimentModel.

Will work the same way as ChurnModel:
- Pass in a DataFrame with a text column and a label column
- It cleans the text, builds TF-IDF features, tests all 3 algorithms, picks the best

Planned usage:
    from sentiment.sentiment_model import SentimentModel

    model = SentimentModel(text_col="review", target_col="sentiment")
    model.fit(df)
    predictions = model.predict(["Great product!", "Terrible experience."])

Algorithm files are ready:
    - xgboost_clf.py
    - random_forest_clf.py
    - logistic_clf.py
"""

# TODO


class SentimentModel:

    def __init__(self, text_col="text", target_col="sentiment"):
        self.text_col   = text_col
        self.target_col = target_col
        raise NotImplementedError("SentimentModel coming soon.")

    def fit(self, df):
        raise NotImplementedError

    def predict(self, texts):
        raise NotImplementedError
