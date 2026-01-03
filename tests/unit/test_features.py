"""Unit tests for feature engineering logic.

These are fast, isolated tests with no external dependencies.
Tests the hashing and embedding logic for feature engineering.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.build_features import build_features, hash_feature


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering logic."""

    def test_hash_feature_basic(self):
        """Test hash_feature returns correct bucket index for known input."""
        # Test with a known string - hash should be deterministic
        value = "test_seller_123"
        n_buckets = 1000
        
        # Hash should return a value between 0 and n_buckets-1
        result = hash_feature(value, n_buckets)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, n_buckets)
        
        # Hash should be deterministic (same input = same output)
        result2 = hash_feature(value, n_buckets)
        self.assertEqual(result, result2)

    def test_hash_feature_different_buckets(self):
        """Test hash_feature with different bucket sizes."""
        value = "test_brand_nike"
        
        # Test with different bucket sizes
        result_100 = hash_feature(value, 100)
        result_1000 = hash_feature(value, 1000)
        
        self.assertGreaterEqual(result_100, 0)
        self.assertLess(result_100, 100)
        self.assertGreaterEqual(result_1000, 0)
        self.assertLess(result_1000, 1000)

    def test_hash_feature_empty_string(self):
        """Test hash_feature handles empty strings."""
        result = hash_feature("", 1000)
        self.assertEqual(result, 0)
        
        # Test with None/NaN
        result_nan = hash_feature(pd.NA, 1000)
        self.assertEqual(result_nan, 0)

    def test_hash_feature_consistency(self):
        """Test hash_feature produces consistent results across multiple calls."""
        test_values = ["seller_1", "seller_2", "brand_nike", "brand_adidas"]
        n_buckets = 1000
        
        # First pass
        results1 = [hash_feature(v, n_buckets) for v in test_values]
        
        # Second pass - should be identical
        results2 = [hash_feature(v, n_buckets) for v in test_values]
        
        self.assertEqual(results1, results2)

    def test_build_features_creates_hashed_features(self):
        """Test build_features creates hashed features correctly."""
        # Create sample data
        data = pd.DataFrame({
            "title": ["Test Product", "Another Product"],
            "seller_id": ["seller_123", "seller_456"],
            "brand": ["Nike", "Adidas"],
            "price": [99.99, 149.99],
            "rating": [4.5, 4.8],
            "reviews_count": [100, 200],
            "subcategory": ["Shoes", "Clothing"]
        })
        
        # Build features
        features = build_features(data)
        
        # Check that hashed features exist
        self.assertIn("seller_id_hashed", features.columns)
        self.assertIn("brand_hashed", features.columns)
        self.assertIn("subcategory_hashed", features.columns)
        
        # Check that hashed features are within expected range
        self.assertTrue(all(0 <= x < 1000 for x in features["seller_id_hashed"]))
        self.assertTrue(all(0 <= x < 1000 for x in features["brand_hashed"]))
        self.assertTrue(all(0 <= x < 1000 for x in features["subcategory_hashed"]))

    def test_build_features_handles_missing_columns(self):
        """Test build_features handles missing columns gracefully."""
        # Create minimal data
        data = pd.DataFrame({
            "title": ["Test Product"],
            "price": [99.99]
        })
        
        # Should not raise an error
        features = build_features(data)
        self.assertIsNotNone(features)
        self.assertGreater(len(features.columns), 0)

    def test_build_features_numeric_output(self):
        """Test build_features returns all numeric features."""
        data = pd.DataFrame({
            "title": ["Test Product"],
            "seller_id": ["seller_123"],
            "brand": ["Nike"],
            "price": [99.99],
            "rating": [4.5],
            "reviews_count": [100],
            "subcategory": ["Shoes"]
        })
        
        features = build_features(data)
        
        # All columns should be numeric (float)
        for col in features.columns:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(features[col]),
                f"Column {col} is not numeric"
            )


if __name__ == "__main__":
    unittest.main()

