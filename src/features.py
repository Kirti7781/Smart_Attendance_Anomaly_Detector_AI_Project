import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Turn raw attendance rows into a numeric feature matrix.

    Ideas: check_in deviation from student's mean, rolling attendance rate,
    absence streak length, weekday, duration z-score.
    """
    # TODO (Nitin): implement
    raise NotImplementedError


def load_data(path: str = "data/attendance.csv"):
    """Return (dataframe, X, y) where y = is_anomaly labels."""
    df = pd.read_csv(path)
    X = build_features(df)
    y = df["is_anomaly"].values
    return df, X, y
