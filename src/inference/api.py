"""FastAPI inference API for product classification."""

import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb  # type: ignore

from src.features.build_features import build_features
from src.inference.drift_detection import AlgorithmicFallback, DriftDetector

app = FastAPI(
    title="Product Classification API",
    description="API for predicting product categories",
    version="1.0.0",
)

# Add CORS middleware to allow requests from browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and label mapping
model = None
label_mapping = None
drift_detector = None
fallback_model = None
reference_data = None  # Store reference data for drift detection


class ProductRequest(BaseModel):
    """Request model for product classification."""

    title: str = Field(..., description="Product title")
    seller_id: Optional[str] = Field(None, description="Seller ID")
    brand: Optional[str] = Field(None, description="Product brand")
    subcategory: Optional[str] = Field(None, description="Product subcategory")
    price: Optional[float] = Field(None, description="Product price")
    rating: Optional[float] = Field(None, description="Product rating")
    reviews_count: Optional[int] = Field(None, description="Number of reviews")


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    category: str = Field(..., description="Predicted category")
    probabilities: dict = Field(..., description="Probability for each category")
    confidence: float = Field(..., description="Confidence score (max probability)")


def load_model(
    model_path: str = "models/model.txt",
    label_mapping_path: str = "models/label_mapping.joblib",
    reference_data_path: Optional[str] = None,
    fallback_model_path: Optional[str] = None,
):
    """
    Load the trained model, label mapping, and initialize drift detection.

    Design Pattern: Drift Detection & Algorithmic Fallback
    """
    global model, label_mapping, drift_detector, fallback_model, reference_data

    try:
        # Load LightGBM model
        if Path(model_path).exists():
            model = lgb.Booster(model_file=model_path)  # type: ignore
        else:
            # Try to load from MLflow
            import mlflow  # type: ignore

            client = mlflow.tracking.MlflowClient()
            try:
                model_version = client.get_latest_versions(
                    "product_classifier", stages=["Production"]
                )[0]
                model_uri = f"models:/product_classifier/{model_version.version}"
                model = mlflow.lightgbm.load_model(model_uri)  # type: ignore
            except Exception:
                # Fallback: try to find any model
                print("Warning: Could not load model from MLflow. Using default path.")
                model = None

        # Load label mapping
        if Path(label_mapping_path).exists():
            label_mapping = joblib.load(label_mapping_path)
        else:
            print("Warning: Label mapping file not found. Using default mapping.")
            label_mapping = None

        # Design Pattern: Drift Detection & Algorithmic Fallback
        print("\n" + "=" * 60)
        print("DESIGN PATTERN: Drift Detection & Algorithmic Fallback")
        print("=" * 60)

        # Load reference data for drift detection
        reference_data = None
        if reference_data_path and Path(reference_data_path).exists():
            reference_data = pd.read_csv(reference_data_path)
            print(f"✓ Loaded reference data: {len(reference_data)} samples")
        else:
            # Try multiple locations for training data
            possible_paths = [
                Path("data/processed/train_features.csv"),
                Path("data/processed/train.csv"),
                Path("data/raw/products.csv"),
            ]

            for train_data_path in possible_paths:
                if train_data_path.exists():
                    try:
                        reference_data = pd.read_csv(train_data_path)
                        # If it's raw data, we need to build features
                        if "train_features" not in str(train_data_path):
                            # This is raw data, we'll use it for concept drift only
                            print(
                                f"ℹ Loaded raw data: {len(reference_data)} samples (will use for concept drift detection)"
                            )
                        else:
                            print(
                                f"✓ Loaded reference data from training: {len(reference_data)} samples"
                            )
                        break
                    except Exception:
                        continue

            if reference_data is None:
                print(
                    "ℹ Info: No reference data found. Drift detection will use confidence-based monitoring only."
                )
                print("   (This is normal - API works fine without reference data)")

        # Initialize drift detector
        if reference_data is not None:
            # Check if reference_data has features (processed) or needs feature engineering
            if (
                "train_features" in str(reference_data_path)
                if reference_data_path
                else False
            ):
                # Already has features
                drift_detector = DriftDetector(
                    reference_data=reference_data, drift_threshold=0.1, window_size=100
                )
            else:
                # Raw data - will use for concept drift only
                drift_detector = DriftDetector(
                    reference_data=None,  # Will use confidence-based detection
                    drift_threshold=0.1,
                    window_size=100,
                )
            print("✓ Drift detector initialized")
        else:
            drift_detector = DriftDetector(drift_threshold=0.1, window_size=100)
            print("✓ Drift detector initialized (confidence-based monitoring)")

        # Initialize fallback model
        fallback_model = AlgorithmicFallback(
            fallback_model_type="random_forest", label_mapping=label_mapping
        )

        # Load or train fallback model
        if fallback_model_path and Path(fallback_model_path).exists():
            fallback_model.load_fallback_model(fallback_model_path)
            print("✓ Fallback model loaded from file")
        else:
            # Try to train fallback model if training data available
            if reference_data is not None and "category" in reference_data.columns:
                try:
                    # Check if data has features or needs feature engineering
                    if (
                        "train_features" in str(reference_data_path)
                        if reference_data_path
                        else False
                    ):
                        # Has features already
                        X_fallback = reference_data.drop(columns=["category"])
                    else:
                        # Need to build features
                        X_fallback = build_features(reference_data)

                    y_fallback = reference_data["category"]
                    fallback_model.train_fallback_model(X_fallback, y_fallback)
                    # Save fallback model
                    fallback_path = Path("models/fallback_model.joblib")
                    fallback_model.save_fallback_model(str(fallback_path))
                    print("✓ Fallback model trained and saved")
                except Exception as e:
                    print(f"ℹ Info: Could not train fallback model: {e}")
                    print("   (API will work with main model only)")
            else:
                print("ℹ Info: Fallback model not available (optional feature)")
                print("   (API works fine with main model only)")

        print("=" * 60)

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        model = None
        label_mapping = None
        drift_detector = None
        fallback_model = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Product Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ProductRequest):
    """
    Predict product category for a single product.

    Design Pattern: Drift Detection & Algorithmic Fallback
    - Detects data drift
    - Falls back to simpler algorithm if drift detected

    Args:
        request: Product information

    Returns:
        Predicted category and probabilities
    """
    if model is None and fallback_model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train a model first."
        )

    try:
        # Convert request to DataFrame
        data = pd.DataFrame(
            [
                {
                    "title": request.title,
                    "seller_id": request.seller_id or "Unknown",
                    "brand": request.brand or "Unknown",
                    "subcategory": request.subcategory or "Unknown",
                    "price": request.price or 0.0,
                    "rating": request.rating or 0.0,
                    "reviews_count": request.reviews_count or 0,
                }
            ]
        )

        # Build features
        features = build_features(data)

        # Design Pattern: Drift Detection
        use_fallback = False

        if drift_detector is not None:
            # Add to drift detection window
            drift_detector.add_request(features)

            # Check for drift
            drift_result = drift_detector.detect_data_drift(features)

            if drift_result.get("drift_detected", False):
                print(
                    f"⚠️  Data drift detected! Score: {drift_result.get('drift_score', 0):.3f}"
                )
                use_fallback = True

        # Make prediction
        if use_fallback and fallback_model is not None:
            # Use fallback model
            print("Using fallback model due to drift detection")
            fallback_pred, fallback_conf = fallback_model.predict_fallback(features)
            predicted_category = str(fallback_pred[0])
            confidence = float(fallback_conf[0])
            probabilities = {predicted_category: confidence}
        elif model is not None:
            # Use main model
            predictions = model.predict(features, num_iteration=model.best_iteration)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))

            # Check concept drift (low confidence)
            if drift_detector is not None:
                concept_drift = drift_detector.detect_concept_drift(
                    predictions, np.array([confidence]), threshold=0.5
                )
                if concept_drift.get("drift_detected", False):
                    print(
                        f"⚠️  Concept drift detected! Avg confidence: {concept_drift.get('avg_confidence', 0):.3f}"
                    )
                    # Use fallback if confidence is too low
                    if fallback_model is not None and confidence < 0.5:
                        print("Using fallback model due to low confidence")
                        fallback_pred, fallback_conf = fallback_model.predict_fallback(
                            features
                        )
                        predicted_category = str(fallback_pred[0])
                        confidence = float(fallback_conf[0])
                        probabilities = {predicted_category: confidence}
                    else:
                        # Use main model but with lower confidence
                        if label_mapping and "idx_to_label" in label_mapping:
                            predicted_category = label_mapping["idx_to_label"][
                                predicted_class_idx
                            ]
                            probabilities = {
                                label_mapping["idx_to_label"][i]: float(prob)
                                for i, prob in enumerate(predictions[0])
                            }
                        else:
                            predicted_category = f"Category_{predicted_class_idx}"
                            probabilities = {
                                f"Category_{i}": float(prob)
                                for i, prob in enumerate(predictions[0])
                            }
                else:
                    # Normal prediction
                    if label_mapping and "idx_to_label" in label_mapping:
                        predicted_category = label_mapping["idx_to_label"][
                            predicted_class_idx
                        ]
                        probabilities = {
                            label_mapping["idx_to_label"][i]: float(prob)
                            for i, prob in enumerate(predictions[0])
                        }
                    else:
                        predicted_category = f"Category_{predicted_class_idx}"
                        probabilities = {
                            f"Category_{i}": float(prob)
                            for i, prob in enumerate(predictions[0])
                        }
            else:
                # No drift detection, use main model
                if label_mapping and "idx_to_label" in label_mapping:
                    predicted_category = label_mapping["idx_to_label"][
                        predicted_class_idx
                    ]
                    probabilities = {
                        label_mapping["idx_to_label"][i]: float(prob)
                        for i, prob in enumerate(predictions[0])
                    }
                else:
                    predicted_category = f"Category_{predicted_class_idx}"
                    probabilities = {
                        f"Category_{i}": float(prob)
                        for i, prob in enumerate(predictions[0])
                    }
        else:
            # Ultimate fallback
            predicted_category = "Unknown"
            confidence = 0.5
            probabilities = {"Unknown": 0.5}

        response = PredictionResponse(
            category=predicted_category,
            probabilities=probabilities,
            confidence=confidence,
        )

        # Add drift info to response (if needed, could extend response model)
        return response

    except Exception as e:
        # If main model fails, try fallback
        if fallback_model is not None:
            try:
                print(f"Main model failed: {e}, trying fallback")
                features = build_features(data)
                fallback_pred, fallback_conf = fallback_model.predict_fallback(features)
                return PredictionResponse(
                    category=str(fallback_pred[0]),
                    probabilities={str(fallback_pred[0]): float(fallback_conf[0])},
                    confidence=float(fallback_conf[0]),
                )
            except Exception as fallback_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction error (fallback also failed): {str(fallback_error)}",
                )
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(requests: List[ProductRequest]):
    """
    Predict product categories for multiple products.

    Args:
        requests: List of product information

    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train a model first."
        )

    try:
        # Convert requests to DataFrame
        data = pd.DataFrame(
            [
                {
                    "title": req.title,
                    "seller_id": req.seller_id or "Unknown",
                    "brand": req.brand or "Unknown",
                    "subcategory": req.subcategory or "Unknown",
                    "price": req.price or 0.0,
                    "rating": req.rating or 0.0,
                    "reviews_count": req.reviews_count or 0,
                }
                for req in requests
            ]
        )

        # Build features
        features = build_features(data)

        # Make predictions
        predictions = model.predict(features, num_iteration=model.best_iteration)
        predicted_class_indices = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        # Map to labels
        results = []
        for idx, (pred_idx, conf) in enumerate(
            zip(predicted_class_indices, confidences)
        ):
            if label_mapping and "idx_to_label" in label_mapping:
                predicted_category = label_mapping["idx_to_label"][pred_idx]
                probabilities = {
                    label_mapping["idx_to_label"][i]: float(prob)
                    for i, prob in enumerate(predictions[idx])
                }
            else:
                predicted_category = f"Category_{pred_idx}"
                probabilities = {
                    f"Category_{i}": float(prob)
                    for i, prob in enumerate(predictions[idx])
                }

            results.append(
                {
                    "category": predicted_category,
                    "probabilities": probabilities,
                    "confidence": float(conf),
                }
            )

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import os
    import sys

    import uvicorn  # type: ignore

    # Use 127.0.0.1 on Windows to avoid WinError 10022
    # For cloud deployment, use 0.0.0.0 and PORT from environment
    host = os.getenv("HOST", "127.0.0.1" if sys.platform == "win32" else "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
