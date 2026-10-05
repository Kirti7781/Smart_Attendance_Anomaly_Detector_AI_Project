from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.features import load_data
from src.model_isolation_forest import IsolationForestDetector
from src.model_autoencoder import AutoencoderDetector
from src.improved_model import HybridDetector


def evaluate_model(detector, X, y) -> dict:
    """Fit detector and return precision, recall, F1, ROC-AUC."""
    # TODO (Rajesh): fit, predict, score -> dict of metrics
    raise NotImplementedError


def main():
    df, X, y = load_data()
    # TODO (Rajesh): run all three models, build comparison table,
    # save to outputs/results.csv and plots (bar chart, ROC curves)
    raise NotImplementedError


if __name__ == "__main__":
    main()
