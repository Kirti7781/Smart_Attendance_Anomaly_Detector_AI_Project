"""Synthetic attendance data generator.

Owner: Arjun Kumar (btech/10433/24)
Output: data/attendance.csv with columns defined in src/base.py (COLUMNS).

Each row = one student on one working day.
  check_in_minute  : minutes after midnight (540 = 9:00 AM), 0 if absent
  duration_minutes : minutes spent in class, 0 if absent
  present          : 1 / 0
  is_anomaly       : ground-truth label (1 = injected anomaly)

Injected anomaly types
  1. Proxy attendance : present, but check-in at a very odd time and/or
     a very short / very long stay.
  2. Absence streak   : a long run (7-10 days) of consecutive absences.
     The first 2 absent days are plausible, so only day 3 onwards of the
     run is labelled anomalous.
"""
import os
import numpy as np
import pandas as pd


def generate_attendance(n_students: int = 60, n_days: int = 90,
                        anomaly_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Create normal attendance and inject labelled anomalies."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=n_days)

    # --- 1. normal behaviour: every student has habits -------------------
    rows = []
    for sid in range(1, n_students + 1):
        habit_checkin = rng.normal(540, 12)      # usual arrival time
        habit_duration = rng.normal(60, 4)       # usual stay (minutes)
        attend_prob = rng.uniform(0.85, 0.97)    # how regular the student is
        for d in dates:
            present = int(rng.random() < attend_prob)
            if present:
                check_in = rng.normal(habit_checkin, 8)
                duration = rng.normal(habit_duration, 3)
            else:
                check_in, duration = 0.0, 0.0
            rows.append([sid, d.date().isoformat(), round(check_in, 1),
                         round(duration, 1), present, 0])
    df = pd.DataFrame(rows, columns=["student_id", "date", "check_in_minute",
                                     "duration_minutes", "present", "is_anomaly"])

    # --- 2. inject anomalies --------------------------------------------
    target = int(anomaly_rate * len(df))
    n_proxy = target // 2

    # (a) proxy attendance on randomly chosen present rows
    present_idx = df.index[df["present"] == 1].to_numpy()
    proxy_idx = rng.choice(present_idx, size=n_proxy, replace=False)
    for i in proxy_idx:
        kind = rng.integers(0, 3)
        if kind == 0:      # arrives far too early / late
            df.at[i, "check_in_minute"] = float(rng.choice([rng.uniform(300, 420),
                                                            rng.uniform(690, 780)]))
        elif kind == 1:    # arrives on time but leaves almost immediately
            df.at[i, "duration_minutes"] = float(rng.uniform(3, 15))
        else:              # both odd
            df.at[i, "check_in_minute"] = float(rng.uniform(300, 420))
            df.at[i, "duration_minutes"] = float(rng.uniform(3, 15))
        df.at[i, "is_anomaly"] = 1

    # (b) absence streaks (7-10 days) until the quota is filled
    n_streak_rows = 0
    while n_streak_rows < target - n_proxy:
        sid = int(rng.integers(1, n_students + 1))
        length = int(rng.integers(7, 11))
        start = int(rng.integers(0, n_days - length))
        sel = df.index[(df["student_id"] == sid)]
        sel = sel[start:start + length]
        df.loc[sel, ["check_in_minute", "duration_minutes", "present"]] = 0
        df.loc[sel[2:], "is_anomaly"] = 1        # label from 3rd absent day
        n_streak_rows += len(sel) - 2

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    data = generate_attendance()
    data.to_csv("data/attendance.csv", index=False)
    print("Saved data/attendance.csv", data.shape,
          "| anomalies:", int(data["is_anomaly"].sum()))
