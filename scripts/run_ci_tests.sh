#!/bin/bash
# Local CI test runner - simulates the CI pipeline locally

set -e  # Exit on error

echo "=========================================="
echo "Running Local CI Pipeline Simulation"
echo "=========================================="

# Step 1: Lint
echo ""
echo "Step 1: Running linters..."
echo "----------------------------------------"
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || {
    echo "✗ Linting failed (syntax errors)"
    exit 1
}
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
pylint src/ --fail-under=6.0 || echo "⚠ Pylint warnings (non-blocking)"

# Step 2: Unit Tests
echo ""
echo "Step 2: Running unit tests..."
echo "----------------------------------------"
pytest tests/unit/ -v --cov=src --cov-report=term-missing || {
    echo "✗ Unit tests failed"
    exit 1
}

# Step 3: Component Tests
echo ""
echo "Step 3: Running component tests..."
echo "----------------------------------------"
pytest tests/component/ -v || {
    echo "✗ Component tests failed"
    exit 1
}

echo ""
echo "=========================================="
echo "✓ All CI tests passed!"
echo "=========================================="

