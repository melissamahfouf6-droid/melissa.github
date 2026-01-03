#!/bin/bash
# Build script for Docker packaging
# "Only build your binaries once" - this script packages the model and serving code

set -e  # Exit on error

echo "=========================================="
echo "Building Docker Image"
echo "=========================================="

# Build the Docker image
docker build -t mlops-product-classifier:latest .

echo "=========================================="
echo "Build completed successfully"
echo "=========================================="

# Optional: Tag for registry
# docker tag mlops-product-classifier:latest your-registry/mlops-product-classifier:latest

