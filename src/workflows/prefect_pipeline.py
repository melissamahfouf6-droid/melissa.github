"""Prefect orchestration pipeline for product classification."""

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from prefect import flow, task  # type: ignore

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Import actual MLflow package
import mlflow  # type: ignore
from src.data.load import load_data
from src.data.preprocess import preprocess_data, split_data
from src.features.build_features import build_features
from src.models.train import evaluate_model, train_model
from src.tracking_utils.tracking import register_model, setup_mlflow


@task(name="load_raw_data", log_prints=True)
def load_raw_data_task(data_path: str = None) -> pd.DataFrame:
    """Load raw data."""
    print("Loading raw data...")
    data = load_data(data_path)
    print(f"Loaded {len(data)} samples")
    return data


@task(name="preprocess_data", log_prints=True)
def preprocess_data_task(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data."""
    print("Preprocessing data...")
    processed_data = preprocess_data(raw_data)
    print(f"Preprocessed {len(processed_data)} samples")
    return processed_data


@task(name="build_features", log_prints=True)
def build_features_task(processed_data: pd.DataFrame) -> pd.DataFrame:
    """Build features."""
    print("Building features...")
    features = build_features(processed_data)
    print(f"Built {features.shape[1]} features")
    return features


@task(name="split_data", log_prints=True)
def split_data_task(
    processed_data: pd.DataFrame, test_size: float = 0.2, random_seed: int = 42
) -> Dict[str, Any]:
    """Split data into train/val/test sets."""
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        processed_data, test_size=test_size, random_seed=random_seed
    )

    # Further split train into train and val
    from sklearn.model_selection import train_test_split

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_seed, stratify=y_train
    )

    print(f"Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return {
        "X_train": X_train_final,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train_final,
        "y_val": y_val,
        "y_test": y_test,
    }


@task(name="train_model", log_prints=True)
def train_model_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Dict[str, Any] = None,
) -> tuple:
    """Train model."""
    print("Training model...")
    model, metrics = train_model(X_train, y_train, X_val, y_val, config=config)
    print(f"Training completed. Accuracy: {metrics.get('train_accuracy', 0):.4f}")
    return model, metrics


@task(name="evaluate_model", log_prints=True)
def evaluate_model_task(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate model."""
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
    return metrics


@task(name="register_model", log_prints=True)
def register_model_task(model_name: str = "product_classifier") -> None:
    """Register model in MLflow."""
    print("Registering model in MLflow...")
    try:
        # Get the latest run
        mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name("product_classification")

        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.train_accuracy DESC"],
                max_results=1,
            )

            if not runs.empty:
                run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{run_id}/model"
                register_model(model_uri, model_name)
                print(f"Model registered: {model_name}")
    except Exception as e:
        print(f"Warning: Could not register model: {e}")


@flow(
    name="product_classification_pipeline", log_prints=True, validate_parameters=False
)
def product_classification_pipeline(
    data_path: str = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    mlflow_tracking_uri: str = "file:./mlruns",
    mlflow_experiment_name: str = "product_classification",
    model_config: Dict[str, Any] = None,
    register_model_flag: bool = True,
):
    """
    Main Prefect pipeline for product classification.

    Pipeline steps:
    1. Load raw data
    2. Preprocess data
    3. Build features
    4. Split data
    5. Train model
    6. Evaluate model
    7. Register model (optional)
    """
    print("Starting product classification pipeline...")

    # Setup MLflow
    setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name)

    # Step 1: Load data
    raw_data = load_raw_data_task(data_path)

    # Step 2: Preprocess
    processed_data = preprocess_data_task(raw_data)

    # Step 3: Build features
    features = build_features_task(processed_data)

    # Combine features with target
    if "category" in processed_data.columns:
        features["category"] = processed_data["category"]

    # Step 4: Split data
    data_splits = split_data_task(
        features, test_size=test_size, random_seed=random_seed
    )

    # Step 5: Train model
    model, train_metrics = train_model_task(
        data_splits["X_train"],
        data_splits["y_train"],
        data_splits["X_val"],
        data_splits["y_val"],
        config=model_config,
    )

    # Step 6: Evaluate model
    test_metrics = evaluate_model_task(
        model, data_splits["X_test"], data_splits["y_test"]
    )

    # Step 7: Register model
    if register_model_flag:
        register_model_task()

    print("Pipeline completed successfully!")

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    # Run the pipeline
    product_classification_pipeline()
