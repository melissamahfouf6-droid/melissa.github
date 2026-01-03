# Homework 2 - MLOps CI/CD Pipeline Submission Guide

This guide explains what has been implemented and how to generate the required deliverables for your homework submission.

## What Has Been Implemented

### Part 1: The Commit Stage (CI) ✅

1. **Version Control Setup**: All assets are in the repository
   - Source code, Dockerfiles, tests, configurations

2. **Automated Unit Testing**: `tests/unit/test_features.py`
   - Fast, isolated tests for `hash_feature()` function
   - No external dependencies (no database/network calls)
   - Tests hash bucket correctness, consistency, and feature engineering

3. **Code Analysis/Linting**: 
   - `.flake8` configuration
   - `.pylintrc` configuration
   - Integrated in GitHub Actions workflow

### Part 2: The Automated Acceptance Gate (CD) ✅

1. **Component/Integration Testing**: `tests/component/test_api_integration.py`
   - Tests interaction between model serving and data sources
   - Uses file system (temporary directories)
   - Verifies feature building with actual DataFrames

2. **Build & Package**: 
   - `scripts/build.sh` (Linux/Mac)
   - `scripts/build.ps1` (Windows)
   - Docker build command packages model and serving code

3. **Smoke Test**: `scripts/smoke_test.py`
   - Spins up container and sends prediction request
   - Verifies service responds with 200 OK
   - Tests `/health` and `/predict` endpoints

### Part 3: The "Stop the Line" Simulation ✅

- Documentation in `CI_CD_PIPELINE.md`
- Demonstration script: `scripts/demonstrate_stop_the_line.py`

## Required Deliverables

### 1. Pipeline Configuration Screenshot

**File**: `.github/workflows/ci-cd.yml`

**How to get screenshot**:
1. Push code to GitHub
2. Go to your repository → Actions tab
3. Click on a workflow run
4. Screenshot showing:
   - `commit-stage` job (Build → Unit Test → Lint)
   - `acceptance-gate` job (Package → Smoke Test)

**What to show**:
- The workflow file structure
- All stages visible in GitHub Actions UI

### 2. Test Results - Evidence A (Success)

**How to generate**:
1. Ensure all tests pass locally:
   ```bash
   pytest tests/unit/ -v
   pytest tests/component/ -v
   ```
2. Push to GitHub
3. Wait for GitHub Actions to complete
4. Screenshot the green checkmarks showing:
   - ✅ commit-stage passed
   - ✅ acceptance-gate passed
   - All test results showing "passed"

### 3. Test Results - Evidence B (Failure/Stop the Line)

**How to demonstrate**:

**Option 1: Break hash_feature function**
1. Edit `src/features/build_features.py`:
   ```python
   def hash_feature(value: str, n_buckets: int = 1000) -> int:
       return 0  # BUG: Always returns 0
   ```
2. Commit and push:
   ```bash
   git add src/features/build_features.py
   git commit -m "Test: Introduce bug for CI/CD demonstration"
   git push
   ```
3. Screenshot the failed pipeline:
   - ❌ commit-stage failed
   - Unit tests failed
   - Pipeline stopped (acceptance-gate didn't run)

**Option 2: Introduce syntax error**
1. Edit `src/features/build_features.py`:
   ```python
   hash_value = int(hashlib.md5(str(value).encode()).hexdigest(), 16
   # Missing closing parenthesis
   ```
2. Commit and push
3. Screenshot showing:
   - ❌ Linting failed (syntax error detected)
   - Pipeline stopped

### 4. Test Code

**Unit Test**: `tests/unit/test_features.py`
- Copy the entire file content
- Explain: "This is a fast unit test because it has no external dependencies, no database/network calls, and tests pure computation (hashing). It runs in milliseconds."

**Smoke Test**: `scripts/smoke_test.py`
- Copy the entire file content
- Explain: "This is an end-to-end test because it tests the complete system: container → API → model → response. It sends real HTTP requests and verifies the entire deployment stack works from a user's perspective."

## Running Tests Locally

### Quick Test Run
```bash
# Run all unit tests
pytest tests/unit/test_features.py -v

# Run component tests
pytest tests/component/test_api_integration.py -v

# Run linting
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pylint src/
```

### Full CI Simulation
```bash
# Linux/Mac
bash scripts/run_ci_tests.sh

# Windows
powershell scripts/run_ci_tests.ps1
```

### Smoke Test (requires running container)
```bash
# 1. Build image
docker build -t mlops-product-classifier:latest .

# 2. Start container
docker run -d -p 8000:8000 --name test-container mlops-product-classifier:latest

# 3. Run smoke test
python scripts/smoke_test.py

# 4. Cleanup
docker stop test-container && docker rm test-container
```

## File Structure

```
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # Pipeline configuration
├── tests/
│   ├── unit/
│   │   └── test_features.py       # Unit tests (fast, isolated)
│   └── component/
│       └── test_api_integration.py # Component tests (with file system)
├── scripts/
│   ├── smoke_test.py              # Smoke test (end-to-end)
│   ├── build.sh                   # Build script (Linux/Mac)
│   ├── build.ps1                  # Build script (Windows)
│   ├── run_ci_tests.sh            # Local CI simulation
│   └── run_ci_tests.ps1           # Local CI simulation (Windows)
├── .flake8                        # Flake8 configuration
├── .pylintrc                      # Pylint configuration
├── CI_CD_PIPELINE.md             # Detailed documentation
└── HOMEWORK_2_SUBMISSION_GUIDE.md # This file
```

## Key Explanations for Report

### Why Unit Test is "Fast"
- **No external dependencies**: No database connections, no network calls
- **Pure computation**: Just hash calculations in memory
- **Isolated**: Each test is independent, no shared state
- **Runs in milliseconds**: Typical execution time < 1 second for all tests

### Why Smoke Test is "End-to-End"
- **Complete system**: Tests container → API → model → response
- **Real HTTP requests**: Actual network calls to verify deployment
- **User perspective**: Tests what a real user would experience
- **Deployment verification**: Proves the entire stack works after deployment

## Troubleshooting

### Tests fail locally
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Check Python version: Should be 3.10+
- Run tests individually to identify specific failures

### GitHub Actions fails
- Check the Actions tab for detailed error messages
- Ensure all required files are committed
- Verify Dockerfile is correct
- Check that model files exist (or API handles missing models gracefully)

### Smoke test fails
- Ensure container is running: `docker ps`
- Check container logs: `docker logs test-container`
- Verify port 8000 is available
- Ensure model files exist in the container

## Next Steps

1. **Test locally**: Run `pytest tests/unit/ -v` to verify unit tests pass
2. **Push to GitHub**: Commit and push all files
3. **Verify pipeline**: Check GitHub Actions tab for green build
4. **Demonstrate failure**: Introduce bug and push to show pipeline blocks
5. **Fix and verify**: Fix bug, push again to show green build
6. **Generate screenshots**: Capture all required evidence
7. **Write report**: Include screenshots and test code with explanations

## Questions?

Refer to `CI_CD_PIPELINE.md` for detailed explanations of each component.

