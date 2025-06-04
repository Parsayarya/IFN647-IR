from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class PRRMModel:
    def __init__(self):
        # Use L2 regularization, balanced class weights, and feature scaling
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=42
            ))
        ])

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        return self.pipeline.predict_proba(X)[:, 1]
