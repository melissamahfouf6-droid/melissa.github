"""Smoke test script for deployment verification.

This script spins up the container and sends a prediction request
to verify the service is up and responding (returning 200 OK).
This is the critical "Deployment Test" for the CI/CD pipeline.
"""

import sys
import time
from pathlib import Path

import requests

# Configuration
API_URL = "http://localhost:8000"
MAX_RETRIES = 30
RETRY_DELAY = 2  # seconds


def check_health():
    """Check if the API health endpoint is responding."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_prediction():
    """Test the prediction endpoint with a sample request."""
    sample_request = {
        "title": "Samsung Galaxy S21 Smartphone",
        "seller_id": "seller_123",
        "brand": "Samsung",
        "subcategory": "Electronics",
        "price": 699.99,
        "rating": 4.5,
        "reviews_count": 1500
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=sample_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Prediction test passed")
            print(f"  Predicted category: {result.get('category', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")
            return True
        else:
            print(f"✗ Prediction test failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Prediction test failed: {e}")
        return False


def wait_for_service(max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Wait for the service to become available."""
    print(f"Waiting for service at {API_URL}...")
    for i in range(max_retries):
        if check_health():
            return True
        if i < max_retries - 1:
            print(f"  Retry {i+1}/{max_retries} in {retry_delay} seconds...")
            time.sleep(retry_delay)
    return False


def main():
    """Run smoke tests."""
    print("=" * 60)
    print("SMOKE TEST: Deployment Verification")
    print("=" * 60)
    
    # Step 1: Wait for service to be ready
    if not wait_for_service():
        print("\n✗ FAILED: Service did not become available")
        sys.exit(1)
    
    # Step 2: Test health endpoint
    print("\n1. Testing health endpoint...")
    if not check_health():
        print("\n✗ FAILED: Health check failed")
        sys.exit(1)
    
    # Step 3: Test prediction endpoint
    print("\n2. Testing prediction endpoint...")
    if not test_prediction():
        print("\n✗ FAILED: Prediction test failed")
        sys.exit(1)
    
    # All tests passed
    print("\n" + "=" * 60)
    print("✓ ALL SMOKE TESTS PASSED")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

