"""Model training with LightGBM and MLflow tracking."""

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb  # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Import actual MLflow package
import mlflow  # type: ignore

# Import design patterns
from src.data.rebalancing import (
    calculate_class_weights,
    check_class_imbalance,
    rebalance_data,
    reframe_problem,
)
from src.models.checkpoints import ModelCheckpoint

# Try to import mlflow.lightgbm, use generic logging if not available
try:
    import mlflow.lightgbm  # type: ignore

    HAS_LIGHTGBM_SUPPORT = True
except ImportError:
    HAS_LIGHTGBM_SUPPORT = False


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    config: Dict[str, Any] = None,
    mlflow_experiment_name: str = "product_classification",
    enable_reframing: bool = True,
    enable_rebalancing: bool = True,
    rebalancing_method: str = "class_weight",
    enable_checkpoints: bool = True,
    checkpoint_dir: str = "models/checkpoints",
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """
    Train LightGBM model for product classification.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        config: Model configuration dictionary
        mlflow_experiment_name: MLflow experiment name
        enable_reframing: Enable reframing pattern (combine minority classes)
        enable_rebalancing: Enable rebalancing pattern (handle class imbalance)
        rebalancing_method: Rebalancing method ('class_weight', 'oversample', 'undersample', 'SMOTE')
        enable_checkpoints: Enable checkpoint pattern (save model during training)
        checkpoint_dir: Directory for checkpoints

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Design Pattern: Reframing & Rebalancing
    print("\n" + "=" * 60)
    print("DESIGN PATTERN: Reframing & Rebalancing")
    print("=" * 60)

    # Check for class imbalance
    imbalance_info = check_class_imbalance(y_train)
    print(f"Class imbalance check:")
    print(f"  Is imbalanced: {imbalance_info['is_imbalanced']}")
    print(f"  Imbalance ratio: {imbalance_info['imbalance_ratio']:.3f}")
    print(f"  Class distribution: {imbalance_info['class_distribution']}")

    # Reframing: Combine minority classes if needed
    if enable_reframing and imbalance_info["is_imbalanced"]:
        X_train, y_train = reframe_problem(X_train, y_train, method="combine_minority")
    # if y_val is not None:
    # Reframe validation set too
    #  _, y_val = reframe_problem(X_val, y_val, method="combine_minority")

    # Rebalancing: Handle class imbalance
    if enable_rebalancing and imbalance_info["is_imbalanced"]:
        X_train, y_train = rebalance_data(
            X_train, y_train, method=rebalancing_method, random_state=42
        )

    # Get number of unique classes and create label mapping (after reframing)
    class_labels = sorted(y_train.unique())
    n_classes = len(class_labels)

    if n_classes < 2:
        raise ValueError(
            f"Need at least 2 classes for classification, found {n_classes}"
        )

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    # Convert string labels to numeric indices (LightGBM requires numeric labels)
    # Ensure we get a numeric Series/array
    y_train_numeric = y_train.map(label_to_idx)
    # Check for any unmapped labels
    if y_train_numeric.isna().any():
        unmapped = y_train[y_train_numeric.isna()].unique()
        raise ValueError(f"Found unmapped labels in training data: {unmapped}")
    y_train_numeric = y_train_numeric.astype(int)

    if y_val is not None:
        y_val_numeric = y_val.map(label_to_idx)
        # Check for any unmapped labels in validation
        if y_val_numeric.isna().any():
            unmapped = y_val[y_val_numeric.isna()].unique()
            raise ValueError(f"Found unmapped labels in validation data: {unmapped}")
        y_val_numeric = y_val_numeric.astype(int)
    else:
        y_val_numeric = None

    if config is None:
        config = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0,
            "random_state": 42,
        }
    else:
        # Ensure num_class is set in config
        if "num_class" not in config:
            config["num_class"] = n_classes
        if "objective" not in config:
            config["objective"] = "multiclass"

    # Design Pattern: Checkpoints
    checkpoint_manager = None
    if enable_checkpoints:
        checkpoint_manager = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            save_freq=10,
            keep_best=True,
            max_checkpoints=5,
        )
        print("\n" + "=" * 60)
        print("DESIGN PATTERN: Checkpoints")
        print("=" * 60)
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Checkpoints enabled: Save every 10 iterations")

    # Calculate class weights for rebalancing
    class_weights = None
    if enable_rebalancing and rebalancing_method == "class_weight":
        class_weights = calculate_class_weights(y_train)
        print(f"\nClass weights: {class_weights}")
        # Convert to LightGBM format (list of weights per class)
        weight_list = [class_weights[label] for label in sorted(class_labels)]
        # LightGBM uses class_weight parameter
        config["class_weight"] = weight_list

    # Set up MLflow
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples", len(X_train))
        mlflow.log_param("n_classes", n_classes)
        mlflow.log_param("reframing_enabled", enable_reframing)
        mlflow.log_param("rebalancing_enabled", enable_rebalancing)
        mlflow.log_param("rebalancing_method", rebalancing_method)
        mlflow.log_param("checkpoints_enabled", enable_checkpoints)

        # Log imbalance info
        mlflow.log_metric("imbalance_ratio", imbalance_info["imbalance_ratio"])
        mlflow.log_param("is_imbalanced", imbalance_info["is_imbalanced"])

        # Prepare data for LightGBM - ensure all features are numeric
        # Convert to numeric and fill any NaN
        X_train_clean = X_train.copy()
        for col in X_train_clean.columns:
            X_train_clean[col] = pd.to_numeric(
                X_train_clean[col], errors="coerce"
            ).fillna(0)
        X_train_clean = X_train_clean.astype(float)

        # Use numeric labels for training - ensure it's a numpy array with int dtype
        y_train_numeric_array = np.array(y_train_numeric, dtype=np.int32)
        train_data = lgb.Dataset(X_train_clean, label=y_train_numeric_array)

        if X_val is not None and y_val_numeric is not None:
            # Clean validation data too
            X_val_clean = X_val.copy()
            for col in X_val_clean.columns:
                X_val_clean[col] = pd.to_numeric(
                    X_val_clean[col], errors="coerce"
                ).fillna(0)
            X_val_clean = X_val_clean.astype(float)

            # Use numeric labels for validation - ensure it's a numpy array with int dtype
            y_val_numeric_array = np.array(y_val_numeric, dtype=np.int32)
            val_data = lgb.Dataset(
                X_val_clean, label=y_val_numeric_array, reference=train_data
            )
            callbacks = [
                lgb.early_stopping(stopping_rounds=10),
                lgb.log_evaluation(period=10),
            ]
        else:
            val_data = None
            callbacks = [lgb.log_evaluation(period=10)]

        # Train model with checkpointing
        model = lgb.train(
            config,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data] + ([val_data] if val_data else []),
            valid_names=["train"] + (["val"] if val_data else []),
            callbacks=callbacks,
            init_model=None,  # Could load from checkpoint here
        )

        # Make predictions (use cleaned data)
        y_train_pred = model.predict(X_train_clean, num_iteration=model.best_iteration)
        y_train_pred_class = np.argmax(y_train_pred, axis=1)

        # Convert predictions back to original labels
        y_train_pred_labels = [idx_to_label[idx] for idx in y_train_pred_class]

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred_labels)
        train_precision = precision_score(
            y_train, y_train_pred_labels, average="weighted", zero_division=0
        )
        train_recall = recall_score(
            y_train, y_train_pred_labels, average="weighted", zero_division=0
        )
        train_f1 = f1_score(
            y_train, y_train_pred_labels, average="weighted", zero_division=0
        )

        metrics = {
            "train_accuracy": train_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # If validation set provided, evaluate on it
        if X_val is not None and y_val_numeric is not None:
            y_val_pred = model.predict(X_val_clean, num_iteration=model.best_iteration)
            y_val_pred_class = np.argmax(y_val_pred, axis=1)
            y_val_pred_labels = [idx_to_label[idx] for idx in y_val_pred_class]

            # Use original y_val (string labels) for metrics
            val_accuracy = accuracy_score(y_val, y_val_pred_labels)
            val_precision = precision_score(
                y_val, y_val_pred_labels, average="weighted", zero_division=0
            )
            val_recall = recall_score(
                y_val, y_val_pred_labels, average="weighted", zero_division=0
            )
            val_f1 = f1_score(
                y_val, y_val_pred_labels, average="weighted", zero_division=0
            )

            val_metrics = {
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            }

            metrics.update(val_metrics)
            mlflow.log_metrics(val_metrics)

        # Save checkpoint after training (with final metrics)
        if enable_checkpoints and checkpoint_manager:
            label_mapping_dict = {
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
            }
            checkpoint_manager.save_checkpoint(
                model=model,
                iteration=model.best_iteration or 100,
                metrics=metrics,
                label_mapping=label_mapping_dict,
                is_best=True,  # Mark as best for now
            )

        # Log model
        if HAS_LIGHTGBM_SUPPORT:
            mlflow.lightgbm.log_model(model, name="model")
        else:
            # Fallback: save model and log as artifact
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.txt")
                model.save_model(model_path)
                mlflow.log_artifact(model_path, artifact_path="model")

        # Save class labels mapping and model locally for API use
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        joblib.dump(
            {"label_to_idx": label_to_idx, "idx_to_label": idx_to_label},
            model_dir / "label_mapping.joblib",
        )

        # Also save model locally for API
        model_path = model_dir / "model.txt"
        model.save_model(str(model_path))
        print(f"Model saved locally to: {model_path}")

        print(f"Model trained successfully!")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Training F1 Score: {train_f1:.4f}")
        if X_val is not None:
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation F1 Score: {val_f1:.4f}")

        return model, metrics


def evaluate_model(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_mapping_path: str = "models/label_mapping.joblib",
) -> Dict[str, float]:
    """
    Evaluate trained model on test set.

    Args:
        model: Trained LightGBM model
        X_test: Test features
        y_test: Test target
        label_mapping_path: Path to label mapping file

    Returns:
        Dictionary of evaluation metrics
    """
    # Load label mapping
    label_mapping = joblib.load(label_mapping_path)
    idx_to_label = label_mapping["idx_to_label"]

    # Make predictions - ensure numeric types
    X_test_clean = X_test.copy()
    for col in X_test_clean.columns:
        X_test_clean[col] = pd.to_numeric(X_test_clean[col], errors="coerce").fillna(0)
    X_test_clean = X_test_clean.astype(float)

    y_pred_proba = model.predict(X_test_clean, num_iteration=model.best_iteration)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    y_pred_labels = [idx_to_label[idx] for idx in y_pred_class]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(
        y_test, y_pred_labels, average="weighted", zero_division=0
    )
    recall = recall_score(y_test, y_pred_labels, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred_labels, average="weighted", zero_division=0)

    metrics = {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
    }

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_labels))

    return metrics
