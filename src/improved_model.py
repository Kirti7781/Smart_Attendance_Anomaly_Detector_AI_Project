import numpy as np
from src.base import BaseDetector


class HybridDetector(BaseDetector):
    name = "Hybrid Ensemble (ours)"

    def __init__(self, detectors, weights=None, threshold_percentile: float = 95):
        self.detectors = detectors            # list of fitted/unfitted BaseDetector
        self.weights = weights
        self.threshold_percentile = threshold_percentile
        self.threshold_ = None

    def fit(self, X: np.ndarray):
        # TODO (Nitin): fit sub-detectors, set threshold_
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        # TODO (Nitin): min-max normalise each score, weighted sum (+ rule score)
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO (Nitin)
        raise NotImplementedError
