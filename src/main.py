"""Main entry point for training pipeline."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

from sklearn.model_selection import train_test_split

import mlflow  # type: ignore

# Import modules
from src.data.load import generate_sample_data, load_data
from src.data.preprocess import preprocess_data, split_data
from src.features.build_features import build_features
from src.models.train import evaluate_model, train_model
from src.tracking_utils.tracking import setup_mlflow


def main():
    """Main function to run the training pipeline."""
    print("=" * 60)
    print("Product Classification Training Pipeline")
    print("=" * 60)

    # Step 1: Setup MLflow (supports cloud via environment variables)
    print("\n[1/7] Setting up MLflow...")
    import os

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    setup_mlflow(tracking_uri, "product_classification")
    print(f"✓ MLflow configured (tracking URI: {tracking_uri})")

    # Step 2: Load data
    print("\n[2/7] Loading data...")
    try:
        data = load_data()
        print(f"✓ Loaded {len(data)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating sample data...")
        data = generate_sample_data(n_samples=10000)
        # Save it
        output_path = project_root / "data" / "raw" / "products.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"✓ Generated and saved {len(data)} samples")

    # Step 3: Preprocess
    print("\n[3/7] Preprocessing data...")
    processed_data = preprocess_data(data)
    print(f"✓ Preprocessed {len(processed_data)} samples")

    # Step 4: Build features
    print("\n[4/7] Building features...")
    features = build_features(processed_data)
    # Add target
    if "category" in processed_data.columns:
        features["category"] = processed_data["category"]
    print(f"✓ Built {features.shape[1]} features")

    # Step 5: Split data
    print("\n[5/7] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        features, test_size=0.2, random_seed=42
    )

    # Further split train into train and val
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"✓ Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 6: Train model
    print("\n[6/7] Training model...")
    print("-" * 60)

    model_config = {
        "objective": "multiclass",
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

    # Enable design patterns
    model, train_metrics = train_model(
        X_train_final,
        y_train_final,
        X_val,
        y_val,
        config=model_config,
        mlflow_experiment_name="product_classification",
        enable_reframing=True,  # Design Pattern: Reframing
        enable_rebalancing=True,  # Design Pattern: Rebalancing
        rebalancing_method="class_weight",  # Options: "class_weight", "oversample", "undersample", "SMOTE"
        enable_checkpoints=True,  # Design Pattern: Checkpoints
        checkpoint_dir="models/checkpoints",
    )

    print("-" * 60)
    print("✓ Model trained successfully!")

    # Step 7: Evaluate on test set
    print("\n[7/7] Evaluating on test set...")
    test_metrics = evaluate_model(model, X_test, y_test)

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - RESULTS")
    print("=" * 60)
    print(f"Train Accuracy:  {train_metrics.get('train_accuracy', 0):.4f}")
    print(f"Train F1 Score:  {train_metrics.get('train_f1', 0):.4f}")
    if "val_accuracy" in train_metrics:
        print(f"Val Accuracy:    {train_metrics.get('val_accuracy', 0):.4f}")
        print(f"Val F1 Score:    {train_metrics.get('val_f1', 0):.4f}")
    print(f"Test Accuracy:   {test_metrics.get('test_accuracy', 0):.4f}")
    print(f"Test F1 Score:   {test_metrics.get('test_f1', 0):.4f}")
    print("=" * 60)
    print("\nTo view results in MLflow UI, run:")
    print(
        "  python -m mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000 --workers 1"
    )
    print("Then open: http://127.0.0.1:5000")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
