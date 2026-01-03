"""Data loading utilities for e-commerce product classification."""

import os
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd


def download_dataset(url: str, output_path: Path) -> bool:
    """
    Download dataset from URL.

    Args:
        url: URL to download from
        output_path: Path where to save the file

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading dataset from {url}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Dataset downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download dataset: {e}")
        return False


def load_data(
    file_path: Optional[str] = None, auto_generate: bool = True
) -> pd.DataFrame:
    """
    Load e-commerce product dataset.

    If file_path is not provided, looks for data in data/raw/ directory.
    If no file exists, generates sample data for testing.

    Args:
        file_path: Path to the CSV file. If None, searches in data/raw/

    Returns:
        DataFrame with product data including: title, seller_id, brand,
        subcategory, price, and category (target)
    """
    if file_path is None:
        # Try to find data in the standard location
        project_root = Path(__file__).parent.parent.parent.parent
        raw_data_dir = project_root / "data" / "raw"
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Look for CSV files in raw data directory
        csv_files = list(raw_data_dir.glob("*.csv"))
        if csv_files:
            file_path = str(csv_files[0])
            print(f"✓ Found existing dataset: {file_path}")
        else:
            # Auto-generate sample data if no file exists
            if auto_generate:
                print("No dataset found. Generating sample data automatically...")
                data = generate_sample_data(n_samples=10000)
                # Save it for future use
                output_path = raw_data_dir / "products.csv"
                data.to_csv(output_path, index=False)
                print(f"✓ Generated and saved {len(data)} samples to {output_path}")
                return data
            else:
                raise FileNotFoundError(
                    f"No dataset found in {raw_data_dir}. Set auto_generate=True to create sample data."
                )

    # Check if file_path is actually a directory
    if os.path.isdir(file_path):
        # If it's a directory, look for CSV files in it
        dir_path = Path(file_path)
        csv_files = list(dir_path.glob("*.csv"))
        if csv_files:
            file_path = str(csv_files[0])
            print(f"✓ Found dataset in directory: {file_path}")
        else:
            # Generate sample data if no CSV files found
            if auto_generate:
                print("No CSV files found in directory. Generating sample data...")
                data = generate_sample_data(n_samples=10000)
                output_path = dir_path / "products.csv"
                data.to_csv(output_path, index=False)
                print(f"✓ Generated and saved {len(data)} samples to {output_path}")
                return data
            else:
                raise FileNotFoundError(f"No CSV files found in {file_path}")

    # Check if file_path is a URL
    if file_path and file_path.startswith(("http://", "https://")):
        # Download from URL
        project_root = Path(__file__).parent.parent.parent.parent
        output_path = project_root / "data" / "raw" / "downloaded_products.csv"
        if download_dataset(file_path, output_path):
            file_path = str(output_path)
        else:
            if auto_generate:
                print("Download failed. Generating sample data instead...")
                data = generate_sample_data(n_samples=10000)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                data.to_csv(output_path.parent / "products.csv", index=False)
                return data
            else:
                raise Exception(f"Failed to download dataset from {file_path}")

    # Now check if it's a valid file
    if os.path.exists(file_path) and os.path.isfile(file_path):
        print(f"✓ Loading dataset from: {file_path}")
        data = pd.read_csv(file_path)
        print(f"✓ Loaded {len(data)} rows, {len(data.columns)} columns")
        return data
    else:
        # Generate sample data if file doesn't exist
        if auto_generate:
            print(f"File not found: {file_path}. Generating sample data...")
            data = generate_sample_data(n_samples=10000)
            # Try to save it where expected
            if file_path:
                try:
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    data.to_csv(file_path, index=False)
                    print(f"✓ Generated and saved to {file_path}")
                except:
                    pass
            return data
        else:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")


def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate sample e-commerce product data for testing.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with synthetic product data
    """
    import numpy as np

    np.random.seed(42)

    # High-cardinality features
    seller_ids = [f"SELLER_{i:05d}" for i in range(1, 5001)]  # 5000 unique sellers
    brands = [
        "Nike",
        "Adidas",
        "Samsung",
        "Apple",
        "Sony",
        "LG",
        "HP",
        "Dell",
        "Canon",
        "Nikon",
        "Microsoft",
        "Google",
        "Amazon",
        "Lenovo",
        "Asus",
        "Acer",
        "Toshiba",
        "Panasonic",
        "Philips",
        "Bosch",
        "Whirlpool",
        "KitchenAid",
        "Dyson",
        "Shark",
        "Bissell",
        "iRobot",
        "Roomba",
        "Fitbit",
        "Garmin",
        "Polar",
        "UnderArmour",
        "Puma",
        "Reebok",
        "NewBalance",
        "Vans",
        "Converse",
        "Timberland",
        "Columbia",
        "NorthFace",
    ]
    subcategories = [
        "Electronics",
        "Clothing",
        "Home & Kitchen",
        "Sports & Outdoors",
        "Books",
        "Toys & Games",
        "Beauty & Personal Care",
        "Automotive",
        "Garden & Tools",
        "Pet Supplies",
        "Baby Products",
        "Office Products",
    ]

    # Product categories (target variable)
    categories = [
        "Electronics",
        "Clothing",
        "Home & Kitchen",
        "Sports & Outdoors",
        "Books",
        "Toys & Games",
        "Beauty",
        "Automotive",
        "Garden",
        "Pet Supplies",
        "Baby",
        "Office",
    ]

    # Generate data
    data = pd.DataFrame(
        {
            "product_id": [f"PROD_{i:06d}" for i in range(1, n_samples + 1)],
            "title": [
                f"{np.random.choice(brands)} {np.random.choice(['Pro', 'Premium', 'Classic', 'Elite', 'Standard'])} "
                f"{np.random.choice(['Product', 'Item', 'Device', 'Tool', 'Accessory'])} "
                f"{np.random.randint(100, 9999)}"
                for _ in range(n_samples)
            ],
            "seller_id": np.random.choice(seller_ids, n_samples),
            "brand": np.random.choice(brands, n_samples),
            "subcategory": np.random.choice(subcategories, n_samples),
            "price": np.random.uniform(9.99, 999.99, n_samples).round(2),
            "rating": np.random.uniform(3.0, 5.0, n_samples).round(1),
            "reviews_count": np.random.randint(0, 10000, n_samples),
        }
    )

    # Create category based on subcategory (with some noise)
    category_mapping = {
        "Electronics": "Electronics",
        "Clothing": "Clothing",
        "Home & Kitchen": "Home & Kitchen",
        "Sports & Outdoors": "Sports & Outdoors",
        "Books": "Books",
        "Toys & Games": "Toys & Games",
        "Beauty & Personal Care": "Beauty",
        "Automotive": "Automotive",
        "Garden & Tools": "Garden",
        "Pet Supplies": "Pet Supplies",
        "Baby Products": "Baby",
        "Office Products": "Office",
    }

    data["category"] = data["subcategory"].map(category_mapping)
    # Add some noise (10% misclassification)
    noise_mask = np.random.random(n_samples) < 0.1
    data.loc[noise_mask, "category"] = np.random.choice(categories, noise_mask.sum())

    return data


def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        data: DataFrame to save
        file_path: Path where to save the file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path, index=False)
