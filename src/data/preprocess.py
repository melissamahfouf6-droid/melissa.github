"""Data preprocessing utilities for e-commerce product classification."""

from typing import Tuple

import pandas as pd


def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess raw product data.

    Args:
        raw_data: Raw DataFrame with product data

    Returns:
        Preprocessed DataFrame
    """
    data = raw_data.copy()

    # Handle missing values
    # Fill missing titles with placeholder
    if "title" in data.columns:
        data["title"] = data["title"].fillna("Unknown Product")

    # Fill missing prices with median
    if "price" in data.columns:
        data["price"] = data["price"].fillna(data["price"].median())

    # Fill missing ratings with median
    if "rating" in data.columns:
        data["rating"] = data["rating"].fillna(data["rating"].median())

    # Fill missing reviews_count with 0
    if "reviews_count" in data.columns:
        data["reviews_count"] = data["reviews_count"].fillna(0)

    # Fill missing seller_id, brand, subcategory with "Unknown"
    for col in ["seller_id", "brand", "subcategory"]:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown")

    # Clean text in title (remove extra spaces, convert to lowercase)
    if "title" in data.columns:
        data["title"] = data["title"].str.lower().str.strip()
        data["title"] = data["title"].str.replace(r"\s+", " ", regex=True)

    # Remove duplicates based on product_id if it exists
    if "product_id" in data.columns:
        data = data.drop_duplicates(subset=["product_id"], keep="first")

    return data


def split_data(
    data: pd.DataFrame,
    target_column: str = "category",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Args:
        data: Preprocessed DataFrame
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    # Separate features and target
    X = data.drop(columns=[target_column], errors="ignore")
    y = data[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    return X_train, X_test, y_train, y_test
