"""Script to demonstrate the "Stop the Line" principle.

This script introduces an intentional bug to show how the CI pipeline
detects failures and blocks deployment.
"""

import sys
from pathlib import Path

# This script is for demonstration purposes only
# It shows what happens when you introduce a bug

print("=" * 60)
print("STOP THE LINE DEMONSTRATION")
print("=" * 60)
print()
print("To demonstrate the 'Stop the Line' principle:")
print()
print("1. Introduce a bug in src/features/build_features.py:")
print("   Change hash_feature() to always return 0:")
print()
print("   def hash_feature(value: str, n_buckets: int = 1000) -> int:")
print("       return 0  # BUG: Should use actual hashing")
print()
print("2. Run the unit tests:")
print("   pytest tests/unit/test_features.py -v")
print()
print("3. The tests will FAIL because:")
print("   - test_hash_feature_consistency expects deterministic hashing")
print("   - test_hash_feature_different_buckets expects different results")
print()
print("4. The CI pipeline will:")
print("   - Detect the test failure")
print("   - Stop the pipeline")
print("   - Block deployment")
print()
print("5. Fix the bug and push again to see green build")
print()
print("=" * 60)
print("Alternative: Introduce a syntax error")
print("=" * 60)
print()
print("Add a syntax error (missing closing parenthesis):")
print()
print("   hash_value = int(hashlib.md5(str(value).encode()).hexdigest(), 16")
print("   # Missing closing parenthesis")
print()
print("This will be caught by:")
print("   - Flake8 (syntax error detection)")
print("   - Python import (build fails)")
print()
print("=" * 60)

