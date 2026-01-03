"""Feature engineering for e-commerce product classification."""

import hashlib
from typing import Any, Dict

import numpy as np
import pandas as pd


def hash_feature(value: str, n_buckets: int = 1000) -> int:
    """
    Hash a categorical value into a fixed number of buckets.

    Args:
        value: Categorical value to hash
        n_buckets: Number of hash buckets

    Returns:
        Hash bucket index (0 to n_buckets-1)
    """
    if pd.isna(value) or value == "":
        return 0

    # Use MD5 hash and take modulo
    hash_value = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
    return hash_value % n_buckets


def build_features(
    data: pd.DataFrame, feature_config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Build features from preprocessed data.

    Implements:
    - Hashed features for high-cardinality categoricals (seller_id, brand)
    - Feature crosses (brand × price_range)
    - Text features from product title
    - Numerical features

    Args:
        data: Preprocessed DataFrame
        feature_config: Configuration dictionary for feature engineering

    Returns:
        DataFrame with engineered features
    """
    if feature_config is None:
        feature_config = {"hash_buckets": 1000, "price_bins": 5, "title_max_words": 10}

    features = pd.DataFrame(index=data.index)

    # 1. Hashed features for high-cardinality categoricals
    if "seller_id" in data.columns:
        features["seller_id_hashed"] = data["seller_id"].apply(
            lambda x: hash_feature(x, feature_config["hash_buckets"])
        )

    if "brand" in data.columns:
        features["brand_hashed"] = data["brand"].apply(
            lambda x: hash_feature(x, feature_config["hash_buckets"])
        )

    # 2. Feature cross: brand × price_range
    if "brand" in data.columns and "price" in data.columns:
        # Create price ranges
        price_bins = pd.cut(
            data["price"],
            bins=feature_config["price_bins"],
            labels=[f"price_range_{i}" for i in range(feature_config["price_bins"])],
        )

        # Create feature cross
        brand_price_cross = data["brand"].astype(str) + "_" + price_bins.astype(str)
        features["brand_price_cross_hashed"] = brand_price_cross.apply(
            lambda x: hash_feature(x, feature_config["hash_buckets"])
        )

        # Also keep price range as separate feature
        price_range_cat = pd.cut(
            data["price"], bins=feature_config["price_bins"], labels=False
        )
        features["price_range"] = (
            pd.to_numeric(price_range_cat, errors="coerce").fillna(0).astype(float)
        )

    # 3. Numerical features
    if "price" in data.columns:
        features["price"] = data["price"]
        features["price_log"] = np.log1p(data["price"])

    if "rating" in data.columns:
        features["rating"] = data["rating"]

    if "reviews_count" in data.columns:
        features["reviews_count"] = data["reviews_count"]
        features["reviews_count_log"] = np.log1p(data["reviews_count"])

    # 4. Text features from title
    if "title" in data.columns:
        # Title length
        features["title_length"] = data["title"].str.len()
        features["title_word_count"] = data["title"].str.split().str.len()

        # Check for common words (simple keyword features)
        keywords = ["pro", "premium", "deluxe", "standard", "basic", "new", "sale"]
        for keyword in keywords:
            features[f"title_has_{keyword}"] = (
                data["title"].str.contains(keyword, case=False, na=False).astype(int)
            )

    # 5. Subcategory encoding (one-hot for low cardinality, hash for high)
    if "subcategory" in data.columns:
        # Use hash encoding since subcategory might have many unique values
        features["subcategory_hashed"] = data["subcategory"].apply(
            lambda x: hash_feature(x, feature_config["hash_buckets"])
        )

    # Fill any remaining NaN values
    features = features.fillna(0)

    # Convert all columns to numeric (LightGBM requires numeric dtypes)
    for col in features.columns:
        if features[col].dtype == "object":
            # Try to convert to numeric
            try:
                features[col] = pd.to_numeric(features[col], errors="coerce")
            except:
                # If conversion fails, use hash encoding
                features[col] = (
                    features[col].astype(str).apply(lambda x: hash_feature(x, 1000))
                )
        # Ensure all are numeric
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)

    # Convert to float for LightGBM compatibility
    features = features.astype(float)

    return features


def get_feature_names() -> list:
    """
    Get list of feature names that will be created.

    Returns:
        List of feature names
    """
    return [
        "seller_id_hashed",
        "brand_hashed",
        "brand_price_cross_hashed",
        "price_range",
        "price",
        "price_log",
        "rating",
        "reviews_count",
        "reviews_count_log",
        "title_length",
        "title_word_count",
        "title_has_pro",
        "title_has_premium",
        "title_has_deluxe",
        "title_has_standard",
        "title_has_basic",
        "title_has_new",
        "title_has_sale",
        "subcategory_hashed",
    ]
