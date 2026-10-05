import numpy as np

# Agreed dataset columns (data/attendance.csv)
COLUMNS = ["student_id", "date", "check_in_minute",
           "duration_minutes", "present", "is_anomaly"]


class BaseDetector:
    """All detectors implement fit / score / predict."""

    name = "base"

    def fit(self, X: np.ndarray):
        """Train on feature matrix X (n_samples, n_features)."""
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly score per row. HIGHER = MORE ANOMALOUS."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 0 (normal) or 1 (anomaly) per row."""
        raise NotImplementedError
