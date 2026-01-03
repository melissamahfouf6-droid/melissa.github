"""MLflow tracking utilities."""

# Import the actual MLflow package (not the local module)
import sys
from typing import Any, Dict, Optional

# CRITICAL: Remove any local mlflow modules to prevent circular imports
# This must happen BEFORE any mlflow import
for key in list(sys.modules.keys()):
    if key.startswith("src.mlflow"):
        del sys.modules[key]

# Also remove mlflow if it's the wrong one
if "mlflow" in sys.modules:
    mlflow_mod = sys.modules["mlflow"]
    # Check if it's the local module (has __file__ pointing to src)
    if (
        hasattr(mlflow_mod, "__file__")
        and mlflow_mod.__file__
        and "src" in str(mlflow_mod.__file__)
    ):
        del sys.modules["mlflow"]

# Now import the actual MLflow package from site-packages
# Direct import should work now
import mlflow  # type: ignore

# Try to import mlflow.lightgbm, fallback to generic logging if not available
try:
    import mlflow.lightgbm  # type: ignore

    HAS_LIGHTGBM_SUPPORT = True
except ImportError:
    HAS_LIGHTGBM_SUPPORT = False


def setup_mlflow(
    tracking_uri: str = None, experiment_name: str = "product_classification_experiment"
) -> None:
    """
    Set up MLflow tracking with cloud support.

    Priority order:
    1. Environment variable MLFLOW_TRACKING_URI (for cloud deployment)
    2. Parameter tracking_uri
    3. Default file-based tracking (file:./mlruns)

    Args:
        tracking_uri: MLflow tracking server URI (optional, uses env var if not provided)
        experiment_name: Name of the MLflow experiment
    """
    import os

    # Priority: env var > parameter > default
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def start_run(run_name: Optional[str] = None):
    """Start an MLflow run."""
    return mlflow.start_run(run_name=run_name)


def log_param(param_name: str, param_value: Any) -> None:
    """Log a parameter to MLflow."""
    mlflow.log_param(param_name, param_value)


def log_params(params: Dict[str, Any]) -> None:
    """Log multiple parameters to MLflow."""
    mlflow.log_params(params)


def log_metric(
    metric_name: str, metric_value: float, step: Optional[int] = None
) -> None:
    """Log a metric to MLflow."""
    mlflow.log_metric(metric_name, metric_value, step=step)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log multiple metrics to MLflow."""
    mlflow.log_metrics(metrics, step=step)


def log_model(model: Any, model_name: str) -> None:
    """Log a model to MLflow."""
    if HAS_LIGHTGBM_SUPPORT:
        mlflow.lightgbm.log_model(model, model_name)
    else:
        # Fallback to generic model logging
        import os
        import tempfile

        import joblib

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")


def end_run() -> None:
    """End the current MLflow run."""
    mlflow.end_run()


def register_model(
    model_uri: str, model_name: str = "product_classifier", stage: str = "Production"
) -> None:
    """
    Register a model in MLflow Model Registry.

    Args:
        model_uri: URI of the model to register
        model_name: Name for the registered model
        stage: Stage to assign (Production, Staging, Archived)
    """
    mlflow.register_model(model_uri, model_name)

    # Transition to specified stage
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=model_name, version=latest_version, stage=stage
    )
