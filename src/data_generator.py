import numpy as np
import pandas as pd


def generate_attendance(n_students: int = 60, n_days: int = 90,
                        anomaly_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Create normal attendance and inject labelled anomalies.

    Anomaly ideas: proxy (check-in at odd time / very short duration),
    sudden absence streak, present only on one-off days.
    """
    # TODO (Kirti): implement generation + anomaly injection
    raise NotImplementedError


if __name__ == "__main__":
    df = generate_attendance()
    df.to_csv("data/attendance.csv", index=False)
    print("Saved data/attendance.csv", df.shape)
