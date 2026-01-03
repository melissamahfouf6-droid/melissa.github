# MLOps CI/CD Pipeline Documentation

This document describes the CI/CD pipeline implementation for the MLOps project, covering the Commit Stage, Automated Acceptance Gate, and Stop the Line simulation.

## Pipeline Overview

The CI/CD pipeline consists of two main stages:

1. **Commit Stage (CI)**: Build → Unit Test → Lint → Component Test
2. **Acceptance Gate (CD)**: Package → Smoke Test

## Part 1: The Commit Stage (Continuous Integration)

### 1.1 Version Control Setup

All assets are stored in the repository:
- Source code (`src/`)
- Dockerfiles (`Dockerfile`, `docker/`)
- Test files (`tests/`)
- Configuration files (`.flake8`, `.pylintrc`)
- Build scripts (`scripts/`)

### 1.2 Automated Unit Testing

**Location**: `tests/unit/test_features.py`

**Purpose**: Fast, isolated tests for feature engineering logic with no external dependencies.

**Key Tests**:
- `test_hash_feature_basic`: Verifies hash function returns correct bucket index
- `test_hash_feature_consistency`: Ensures deterministic hashing
- `test_build_features_creates_hashed_features`: Tests feature engineering pipeline

**Why it's "fast"**: 
- No database connections
- No network calls
- No file I/O (uses in-memory DataFrames)
- Pure computation tests
- Runs in milliseconds

**Run locally**:
```bash
pytest tests/unit/test_features.py -v
```

### 1.3 Code Analysis/Linting

**Tools**: Flake8 and Pylint

**Configuration**:
- `.flake8`: Flake8 configuration with max line length 100, complexity checks
- `.pylintrc`: Pylint configuration with appropriate thresholds

**Failure Threshold**: 
- Flake8: Syntax errors and undefined names fail the build
- Pylint: Warnings shown but don't fail build (fail-under=6.0)

**Run locally**:
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pylint src/ --fail-under=6.0
```

## Part 2: The Automated Acceptance Gate (CD)

### 2.1 Component/Integration Testing

**Location**: `tests/component/test_api_integration.py`

**Purpose**: Verifies interaction between model serving logic and data sources.

**Key Tests**:
- `test_build_features_integration`: Tests feature building with actual DataFrame (data source)
- `test_feature_consistency_with_api_format`: Ensures features match API expectations
- `test_model_loading_with_file_system`: Tests file system interaction

**Why it's "component" test**:
- Involves file system (temporary directories)
- Tests integration between components (data → features → model)
- Can use mock data sources
- Verifies data consistency

**Run locally**:
```bash
pytest tests/component/test_api_integration.py -v
```

### 2.2 Build & Package

**Scripts**: 
- `scripts/build.sh` (Linux/Mac)
- `scripts/build.ps1` (Windows)

**Purpose**: Packages model and serving code into a deployable Docker container.

**Principle**: "Only build your binaries once" - single Docker build creates the artifact.

**Run locally**:
```bash
# Linux/Mac
bash scripts/build.sh

# Windows
powershell scripts/build.ps1
```

### 2.3 Smoke Test

**Location**: `scripts/smoke_test.py`

**Purpose**: Spins up container and sends a prediction request to verify service is up and responding (200 OK).

**What it does**:
1. Waits for service to be ready (health check)
2. Tests `/health` endpoint
3. Tests `/predict` endpoint with sample data
4. Verifies 200 OK response

**Why it's "end-to-end"**:
- Tests the complete system (container → API → model → response)
- Verifies deployment works from user perspective
- Tests actual HTTP requests
- Validates the entire stack is functional

**Run locally** (after starting container):
```bash
# Start container first
docker run -d -p 8000:8000 --name test-container mlops-product-classifier:latest

# Run smoke test
python scripts/smoke_test.py
```

## Part 3: The "Stop the Line" Simulation

### How to Demonstrate

1. **Introduce a bug** in `src/features/build_features.py`:
   ```python
   # Sabotage: Change hash function to always return 0
   def hash_feature(value: str, n_buckets: int = 1000) -> int:
       return 0  # BUG: Should use actual hashing
   ```

2. **Commit the broken code**:
   ```bash
   git add src/features/build_features.py
   git commit -m "Test: Introduce bug for CI/CD demonstration"
   git push
   ```

3. **Observe pipeline failure**:
   - Unit tests will fail (hash_feature tests expect correct behavior)
   - Pipeline stops at Commit Stage
   - No deployment occurs

4. **Fix the bug** and push again to see green build

### Alternative: Syntax Error

Introduce a syntax error:
```python
def hash_feature(value: str, n_buckets: int = 1000) -> int:
    if pd.isna(value) or value == "":
        return 0
    # Missing closing parenthesis
    hash_value = int(hashlib.md5(str(value).encode()).hexdigest(), 16
    return hash_value % n_buckets
```

This will be caught by:
- Linter (flake8/pylint) - syntax error detection
- Build stage - Python import fails

## Pipeline Configuration

**Location**: `.github/workflows/ci-cd.yml`

**Stages**:
1. **commit-stage**: 
   - Checkout code
   - Install dependencies
   - Lint (flake8 + pylint)
   - Unit tests
   - Component tests

2. **acceptance-gate** (runs only if commit-stage passes):
   - Build Docker image
   - Start container
   - Wait for service
   - Run smoke test
   - Save image artifact

## Running Tests Locally

### All Unit Tests
```bash
pytest tests/unit/ -v
```

### All Component Tests
```bash
pytest tests/component/ -v
```

### All Tests with Coverage
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Linting
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pylint src/
```

### Full Pipeline Simulation
```bash
# 1. Lint
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# 2. Unit tests
pytest tests/unit/ -v

# 3. Component tests
pytest tests/component/ -v

# 4. Build
docker build -t mlops-product-classifier:latest .

# 5. Start container
docker run -d -p 8000:8000 --name test-container mlops-product-classifier:latest

# 6. Smoke test
python scripts/smoke_test.py

# 7. Cleanup
docker stop test-container && docker rm test-container
```

## Test Explanations

### Unit Test (Fast)
**File**: `tests/unit/test_features.py`

**Why it's "fast"**:
- No external dependencies (no database, no network)
- Pure computation
- In-memory operations only
- Runs in < 1 second

**Example**: `test_hash_feature_basic` tests that `hash_feature("test", 1000)` returns a value between 0-999. This is fast because it's just a hash calculation.

### Smoke Test (End-to-End)
**File**: `scripts/smoke_test.py`

**Why it's "end-to-end"**:
- Tests complete system: Container → API → Model → Response
- Real HTTP requests
- Actual deployment verification
- User perspective testing

**Example**: Sends HTTP POST to `/predict` endpoint and verifies 200 OK response with valid prediction. This is end-to-end because it tests the entire stack from HTTP request to model prediction.

## Evidence for Submission

1. **Pipeline Configuration**: `.github/workflows/ci-cd.yml` (screenshot in GitHub Actions)
2. **Test Results (Success)**: Screenshot of green build in GitHub Actions
3. **Test Results (Failure)**: Screenshot of failed build after introducing bug
4. **Test Code**: 
   - Unit Test: `tests/unit/test_features.py`
   - Smoke Test: `scripts/smoke_test.py`

