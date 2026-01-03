# Build script for Docker packaging (PowerShell version)
# "Only build your binaries once" - this script packages the model and serving code

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Building Docker Image" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Build the Docker image
docker build -t mlops-product-classifier:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "Build completed successfully" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
} else {
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "Build failed" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    exit 1
}

# Optional: Tag for registry
# docker tag mlops-product-classifier:latest your-registry/mlops-product-classifier:latest

