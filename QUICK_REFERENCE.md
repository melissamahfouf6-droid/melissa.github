# CI/CD Pipeline Quick Reference

## Files Created

### Tests
- `tests/unit/test_features.py` - Unit tests for feature engineering (FAST, isolated)
- `tests/component/test_api_integration.py` - Component tests (with file system)

### Scripts
- `scripts/smoke_test.py` - End-to-end deployment test
- `scripts/build.sh` / `scripts/build.ps1` - Docker build scripts
- `scripts/run_ci_tests.sh` / `scripts/run_ci_tests.ps1` - Local CI simulation

### Configuration
- `.github/workflows/ci-cd.yml` - GitHub Actions pipeline
- `.flake8` - Flake8 linting configuration
- `.pylintrc` - Pylint configuration

### Documentation
- `CI_CD_PIPELINE.md` - Detailed pipeline documentation
- `HOMEWORK_2_SUBMISSION_GUIDE.md` - Submission guide

## Quick Commands

```bash
# Run unit tests
pytest tests/unit/test_features.py -v

# Run component tests
pytest tests/component/test_api_integration.py -v

# Run linting
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pylint src/

# Build Docker image
docker build -t mlops-product-classifier:latest .

# Run smoke test (after starting container)
python scripts/smoke_test.py

# Run full CI locally
bash scripts/run_ci_tests.sh  # Linux/Mac
powershell scripts/run_ci_tests.ps1  # Windows
```

## Pipeline Stages

1. **Commit Stage** (CI)
   - Lint (flake8 + pylint)
   - Unit tests
   - Component tests

2. **Acceptance Gate** (CD)
   - Build Docker image
   - Start container
   - Smoke test

## Demonstrate "Stop the Line"

1. Break `hash_feature()` in `src/features/build_features.py`:
   ```python
   def hash_feature(value: str, n_buckets: int = 1000) -> int:
       return 0  # BUG
   ```

2. Commit and push
3. Pipeline fails at unit tests
4. Fix and push again → green build

