import numpy as np
from sklearn.ensemble import IsolationForest
from src.base import BaseDetector


class IsolationForestDetector(BaseDetector):
    name = "Isolation Forest"

    def __init__(self, contamination: float = 0.05, seed: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=seed)

    def fit(self, X: np.ndarray):
        # TODO (Alok): fit the model (add scaling if needed)
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        # TODO (Alok): return -self.model.score_samples(X)  (higher = more anomalous)
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO (Alok): convert sklearn's -1/1 output to 1/0
        raise NotImplementedError
