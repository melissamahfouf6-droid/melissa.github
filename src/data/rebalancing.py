"""Reframing & Rebalancing pattern for handling class imbalance."""

from collections import Counter
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def check_class_imbalance(y: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Check if classes are imbalanced.

    Args:
        y: Target labels
        threshold: Imbalance threshold (ratio of minority to majority class)

    Returns:
        Dictionary with imbalance status and statistics
    """
    class_counts = Counter(y)
    total_samples = len(y)
    n_classes = len(class_counts)

    if n_classes == 0:
        return {
            "is_imbalanced": False,
            "imbalance_ratio": 1.0,
            "class_distribution": {},
            "majority_class": None,
            "minority_class": None,
        }

    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = min_count / max_count if max_count > 0 else 0.0

    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    is_imbalanced = imbalance_ratio < threshold

    return {
        "is_imbalanced": is_imbalanced,
        "imbalance_ratio": imbalance_ratio,
        "class_distribution": dict(class_counts),
        "majority_class": majority_class,
        "minority_class": minority_class,
        "n_classes": n_classes,
        "total_samples": total_samples,
    }


def reframe_problem(
    X: pd.DataFrame, y: pd.Series, method: str = "combine_minority"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Reframe the problem by combining minority classes.

    Design Pattern: Reframing
    - Combines rare classes into "Other" category
    - Reduces number of classes
    - Helps with class imbalance

    Args:
        X: Features
        y: Target labels
        method: Reframing method ('combine_minority' or 'binary')

    Returns:
        Reframed (X, y)
    """
    if method == "combine_minority":
        class_counts = Counter(y)
        total_samples = len(y)
        threshold = total_samples * 0.05  # 5% threshold

        # Identify minority classes
        minority_classes = [
            cls for cls, count in class_counts.items() if count < threshold
        ]

        if len(minority_classes) > 0:
            # Combine minority classes into "Other"
            y_reframed = y.copy()
            y_reframed = y_reframed.replace(minority_classes, "Other")
            print(
                f"Reframing: Combined {len(minority_classes)} minority classes into 'Other'"
            )
            return X, y_reframed
        else:
            print("Reframing: No minority classes to combine")
            return X, y

    elif method == "binary":
        # Convert to binary classification (most common vs rest)
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        y_binary = (y == majority_class).astype(int)
        y_binary = y_binary.replace({1: f"{majority_class}", 0: "Other"})
        print(
            f"Reframing: Converted to binary classification ({majority_class} vs Other)"
        )
        return X, y_binary

    else:
        return X, y


def rebalance_data(
    X: pd.DataFrame, y: pd.Series, method: str = "class_weight", random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Rebalance imbalanced dataset.

    Design Pattern: Rebalancing
    - Handles class imbalance in training data
    - Methods: class_weight, SMOTE, undersampling, oversampling

    Args:
        X: Features
        y: Target labels
        method: Rebalancing method
        random_state: Random seed

    Returns:
        Rebalanced (X, y)
    """
    imbalance_info = check_class_imbalance(y)

    if not imbalance_info["is_imbalanced"]:
        print("Rebalancing: Classes are balanced, no rebalancing needed")
        return X, y

    print(
        f"Rebalancing: Imbalance detected (ratio: {imbalance_info['imbalance_ratio']:.3f})"
    )
    print(
        f"  Majority class: {imbalance_info['majority_class']} ({imbalance_info['class_distribution'][imbalance_info['majority_class']]} samples)"
    )
    print(
        f"  Minority class: {imbalance_info['minority_class']} ({imbalance_info['class_distribution'][imbalance_info['minority_class']]} samples)"
    )

    if method == "class_weight":
        # Return original data - class weights will be applied during training
        print(
            "Rebalancing: Using class_weight method (weights applied during training)"
        )
        return X, y

    elif method == "oversample":
        # Oversample minority classes
        from sklearn.utils import resample

        class_counts = Counter(y)
        max_count = max(class_counts.values())

        X_resampled = []
        y_resampled = []

        for cls in class_counts.keys():
            cls_mask = y == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]

            if len(X_cls) < max_count:
                # Oversample
                X_cls_resampled, y_cls_resampled = resample(
                    X_cls,
                    y_cls,
                    replace=True,
                    n_samples=max_count,
                    random_state=random_state,
                )
                print(f"  Oversampled {cls}: {len(X_cls)} -> {len(X_cls_resampled)}")
            else:
                X_cls_resampled, y_cls_resampled = X_cls, y_cls

            X_resampled.append(X_cls_resampled)
            y_resampled.append(y_cls_resampled)

        X_balanced = pd.concat(X_resampled, ignore_index=True)
        y_balanced = pd.concat(y_resampled, ignore_index=True)

        # Shuffle
        indices = np.random.RandomState(random_state).permutation(len(X_balanced))
        X_balanced = X_balanced.iloc[indices].reset_index(drop=True)
        y_balanced = y_balanced.iloc[indices].reset_index(drop=True)

        print(
            f"Rebalancing: Oversampling complete ({len(X)} -> {len(X_balanced)} samples)"
        )
        return X_balanced, y_balanced

    elif method == "undersample":
        # Undersample majority classes
        from sklearn.utils import resample

        class_counts = Counter(y)
        min_count = min(class_counts.values())

        X_resampled = []
        y_resampled = []

        for cls in class_counts.keys():
            cls_mask = y == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]

            if len(X_cls) > min_count:
                # Undersample
                X_cls_resampled, y_cls_resampled = resample(
                    X_cls,
                    y_cls,
                    replace=False,
                    n_samples=min_count,
                    random_state=random_state,
                )
                print(f"  Undersampled {cls}: {len(X_cls)} -> {len(X_cls_resampled)}")
            else:
                X_cls_resampled, y_cls_resampled = X_cls, y_cls

            X_resampled.append(X_cls_resampled)
            y_resampled.append(y_cls_resampled)

        X_balanced = pd.concat(X_resampled, ignore_index=True)
        y_balanced = pd.concat(y_resampled, ignore_index=True)

        # Shuffle
        indices = np.random.RandomState(random_state).permutation(len(X_balanced))
        X_balanced = X_balanced.iloc[indices].reset_index(drop=True)
        y_balanced = y_balanced.iloc[indices].reset_index(drop=True)

        print(
            f"Rebalancing: Undersampling complete ({len(X)} -> {len(X_balanced)} samples)"
        )
        return X_balanced, y_balanced

    elif method == "SMOTE":
        # Use SMOTE for oversampling (requires imbalanced-learn)
        try:
            from imblearn.over_sampling import SMOTE

            # Convert to numeric for SMOTE
            label_to_idx = {label: idx for idx, label in enumerate(sorted(y.unique()))}
            y_numeric = y.map(label_to_idx)

            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled_numeric = smote.fit_resample(X, y_numeric)

            # Convert back to labels
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}
            y_resampled = pd.Series([idx_to_label[idx] for idx in y_resampled_numeric])
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

            print(
                f"Rebalancing: SMOTE complete ({len(X)} -> {len(X_resampled)} samples)"
            )
            return X_resampled, y_resampled
        except ImportError:
            print(
                "Warning: imbalanced-learn not installed, falling back to class_weight"
            )
            return X, y

    else:
        print(f"Rebalancing: Unknown method '{method}', returning original data")
        return X, y


def calculate_class_weights(y: pd.Series) -> Dict[str, float]:
    """
    Calculate class weights for imbalanced data.

    Args:
        y: Target labels

    Returns:
        Dictionary mapping class to weight
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = sorted(y.unique())
    class_weights = compute_class_weight("balanced", classes=np.array(classes), y=y)

    weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    return weight_dict
