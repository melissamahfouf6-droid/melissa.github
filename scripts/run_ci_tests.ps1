# Local CI test runner - simulates the CI pipeline locally (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Running Local CI Pipeline Simulation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Lint
Write-Host ""
Write-Host "Step 1: Running linters..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Linting failed (syntax errors)" -ForegroundColor Red
    exit 1
}
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
pylint src/ --fail-under=6.0
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Pylint warnings (non-blocking)" -ForegroundColor Yellow
}

# Step 2: Unit Tests
Write-Host ""
Write-Host "Step 2: Running unit tests..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
pytest tests/unit/ -v --cov=src --cov-report=term-missing
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Unit tests failed" -ForegroundColor Red
    exit 1
}

# Step 3: Component Tests
Write-Host ""
Write-Host "Step 3: Running component tests..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
pytest tests/component/ -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Component tests failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✓ All CI tests passed!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

