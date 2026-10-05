# Smart Attendance Anomaly Detector

AI course group assignment. Detects suspicious attendance records
(proxy attendance, absence streaks, odd check-in times).

## Team
| Name | Roll No. | Files |
|---|---|---|
| Kirti Singh | btech/10688/24 | src/data_generator.py, app.py |
| Nitin Raj | btech/10503/24 | src/features.py, src/improved_model.py |
| Alok Kumar Choudhary | btech/10684/24 | src/model_isolation_forest.py |
| Arjun Kumar | btech/10433/24 | src/model_autoencoder.py |
| Rajesh Ranjan | btech/10496/24 | src/evaluate.py, notebooks/experiments.ipynb, README.md |

## Problem Formulation
TODO (Rajesh)

## Approaches Compared
1. Isolation Forest (Alok)
2. Autoencoder, reconstruction error (Arjun)
3. Student-designed improvement: hybrid ensemble (Nitin)

## How to Run
```bash
pip install -r requirements.txt
python -m src.data_generator        # creates data/attendance.csv
python -m src.evaluate              # trains all models, writes outputs/
streamlit run app.py                # demo
```

## Results
TODO (Rajesh) - add metrics table and plots from outputs/

## Git Workflow
Each member works on `feature/<name>` and edits only their own files.
