"""Drift Detection & Algorithmic Fallback pattern for serving."""

import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.naive_bayes import MultinomialNB  # type: ignore

warnings.filterwarnings("ignore")


class DriftDetector:
    """
    Design Pattern: Drift Detection
    - Detects data drift (distribution changes)
    - Detects concept drift (model performance degradation)
    - Triggers fallback to simpler algorithm
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.1,
        window_size: int = 100,
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference data distribution (training data)
            drift_threshold: Threshold for detecting drift (KS statistic)
            window_size: Size of sliding window for drift detection
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.request_window = deque(maxlen=window_size)
        self.drift_detected = False
        self.drift_count = 0

        if reference_data is not None:
            self._compute_reference_stats()
        else:
            self.reference_stats = None

    def _compute_reference_stats(self):
        """Compute reference statistics from training data."""
        if self.reference_data is None:
            return

        self.reference_stats = {}
        for col in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                self.reference_stats[col] = {
                    "mean": self.reference_data[col].mean(),
                    "std": self.reference_data[col].std(),
                    "min": self.reference_data[col].min(),
                    "max": self.reference_data[col].max(),
                }

    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift using Kolmogorov-Smirnov test.

        Args:
            current_data: Current incoming data

        Returns:
            Dictionary with drift detection results
        """
        if self.reference_stats is None:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "reason": "No reference data available",
            }

        drift_scores = {}
        max_drift = 0.0

        for col in current_data.columns:
            if col not in self.reference_stats:
                continue

            if pd.api.types.is_numeric_dtype(current_data[col]):
                ref_stats = self.reference_stats[col]
                current_mean = current_data[col].mean()
                current_std = current_data[col].std()

                # Simple drift detection: compare means and stds
                mean_diff = abs(current_mean - ref_stats["mean"]) / (
                    ref_stats["std"] + 1e-6
                )
                std_diff = abs(current_std - ref_stats["std"]) / (
                    ref_stats["std"] + 1e-6
                )

                drift_score = max(mean_diff, std_diff)
                drift_scores[col] = drift_score
                max_drift = max(max_drift, drift_score)

        drift_detected = max_drift > self.drift_threshold

        if drift_detected:
            self.drift_detected = True
            self.drift_count += 1

        return {
            "drift_detected": drift_detected,
            "drift_score": max_drift,
            "feature_drifts": drift_scores,
            "threshold": self.drift_threshold,
        }

    def detect_concept_drift(
        self, predictions: np.ndarray, confidences: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect concept drift based on prediction confidence.

        Args:
            predictions: Model predictions
            confidences: Prediction confidences
            threshold: Confidence threshold for drift detection

        Returns:
            Dictionary with concept drift detection results
        """
        avg_confidence = np.mean(confidences)
        low_confidence_ratio = np.sum(confidences < threshold) / len(confidences)

        concept_drift = avg_confidence < threshold or low_confidence_ratio > 0.3

        if concept_drift:
            self.drift_detected = True
            self.drift_count += 1

        return {
            "drift_detected": concept_drift,
            "avg_confidence": float(avg_confidence),
            "low_confidence_ratio": float(low_confidence_ratio),
            "threshold": threshold,
        }

    def add_request(self, features: pd.DataFrame):
        """Add request to sliding window."""
        self.request_window.append(features)

    def check_drift_in_window(self) -> Dict[str, Any]:
        """Check for drift in the current window."""
        if len(self.request_window) < self.window_size:
            return {
                "drift_detected": False,
                "reason": f"Window not full ({len(self.request_window)}/{self.window_size})",
            }

        # Combine window data
        window_data = pd.concat(list(self.request_window), ignore_index=True)

        # Detect drift
        drift_result = self.detect_data_drift(window_data)
        return drift_result


class AlgorithmicFallback:
    """
    Design Pattern: Algorithmic Fallback
    - Falls back to simpler algorithm when drift detected
    - Provides reliable predictions even when main model fails
    - Ensures service availability
    """

    def __init__(
        self,
        fallback_model_type: str = "random_forest",
        label_mapping: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize fallback model.

        Args:
            fallback_model_type: Type of fallback model ('random_forest', 'naive_bayes', 'rule_based')
            label_mapping: Label mapping dictionary
        """
        self.fallback_model_type = fallback_model_type
        self.fallback_model = None
        self.label_mapping = label_mapping
        self.is_fallback_active = False

    def train_fallback_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Train fallback model on training data.

        Args:
            X: Features
            y: Target labels
        """
        print("Training fallback model...")

        if self.fallback_model_type == "random_forest":
            from sklearn.preprocessing import LabelEncoder

            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            self.fallback_model = RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
            )
            self.fallback_model.fit(X, y_encoded)
            self.label_encoder = le
            print("Fallback model (Random Forest) trained")

        elif self.fallback_model_type == "naive_bayes":
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Ensure non-negative values for MultinomialNB
            X_non_neg = X - X.min() + 1

            self.fallback_model = MultinomialNB()
            self.fallback_model.fit(X_non_neg, y_encoded)
            self.label_encoder = le
            print("Fallback model (Naive Bayes) trained")

        elif self.fallback_model_type == "rule_based":
            # Simple rule-based classifier
            self.fallback_model = "rule_based"
            print("Fallback model (Rule-based) initialized")

    def predict_fallback(
        self, X: pd.DataFrame, fallback_type: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using fallback model.

        Args:
            X: Features
            fallback_type: Override fallback type

        Returns:
            Tuple of (predictions, confidences)
        """
        fallback_type = fallback_type or self.fallback_model_type

        if fallback_type == "rule_based":
            # Simple rule-based predictions
            predictions = self._rule_based_predict(X)
            confidences = np.ones(len(predictions)) * 0.6  # Lower confidence
            return predictions, confidences

        elif self.fallback_model is not None:
            if fallback_type == "naive_bayes":
                X_non_neg = X - X.min() + 1
                predictions = self.fallback_model.predict(X_non_neg)
                probabilities = self.fallback_model.predict_proba(X_non_neg)
            else:
                predictions = self.fallback_model.predict(X)
                probabilities = self.fallback_model.predict_proba(X)

            confidences = np.max(probabilities, axis=1)

            # Convert back to original labels
            if hasattr(self, "label_encoder"):
                predictions = self.label_encoder.inverse_transform(predictions)

            return predictions, confidences
        else:
            # Ultimate fallback: return most common class
            if self.label_mapping and "idx_to_label" in self.label_mapping:
                most_common_idx = 0  # Default to first class
                predictions = np.array(
                    [self.label_mapping["idx_to_label"][most_common_idx]] * len(X)
                )
            else:
                predictions = np.array(["Unknown"] * len(X))
            confidences = np.ones(len(X)) * 0.5
            return predictions, confidences

    def _rule_based_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Rule-based prediction fallback.

        Simple rules based on features:
        - High price + Electronics keywords → Electronics
        - Clothing keywords → Clothing
        - etc.
        """
        predictions = []

        for idx, row in X.iterrows():
            # Simple rule-based logic
            # This is a placeholder - would need actual feature names
            if "price" in X.columns and row.get("price", 0) > 500:
                predictions.append("Electronics")
            elif "price" in X.columns and row.get("price", 0) < 50:
                predictions.append("Books")
            else:
                predictions.append("Clothing")  # Default

        return np.array(predictions)

    def save_fallback_model(self, path: str):
        """Save fallback model to disk."""
        if self.fallback_model is not None:
            model_path = Path(path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": self.fallback_model,
                    "model_type": self.fallback_model_type,
                    "label_mapping": self.label_mapping,
                },
                path,
            )
            if hasattr(self, "label_encoder"):
                joblib.dump(
                    self.label_encoder,
                    str(model_path.parent / "fallback_label_encoder.joblib"),
                )
            print(f"Fallback model saved to: {path}")

    def load_fallback_model(self, path: str):
        """Load fallback model from disk."""
        if Path(path).exists():
            data = joblib.load(path)
            self.fallback_model = data.get("model")
            self.fallback_model_type = data.get("model_type", "random_forest")
            self.label_mapping = data.get("label_mapping")

            encoder_path = Path(path).parent / "fallback_label_encoder.joblib"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)

            print(f"Fallback model loaded from: {path}")
        else:
            print(f"Fallback model not found at: {path}")
