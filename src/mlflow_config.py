"""
MLflow setup helper for electricity price forecasting experiments.
"""
import os
from pathlib import Path

# Project root is one level above src/
_PROJECT_ROOT = Path(__file__).parent.parent

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{_PROJECT_ROOT / 'mlruns.db'}"
)

EXPERIMENT_NAME = "electricity-price-forecasting"


def setup_mlflow():
    """
    Set the MLflow tracking URI and ensure the experiment exists.
    Returns the MLflow experiment object.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    return experiment
