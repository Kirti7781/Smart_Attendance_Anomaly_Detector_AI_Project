import numpy as np
from sklearn.neural_network import MLPRegressor
from src.base import BaseDetector


class AutoencoderDetector(BaseDetector):
    name = "Autoencoder"

    def __init__(self, hidden=(8, 3, 8), threshold_percentile: float = 95, seed: int = 42):
        self.model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=500, random_state=seed)
        self.threshold_percentile = threshold_percentile
        self.threshold_ = None

    def fit(self, X: np.ndarray):
        # TODO (Arjun): scale X, train model on (X, X), set self.threshold_
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        # TODO (Arjun): per-row mean squared reconstruction error
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO (Arjun): 1 if score > self.threshold_ else 0
        raise NotImplementedError
