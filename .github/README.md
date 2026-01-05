# GitHub Actions CI/CD Documentation

This directory contains GitHub Actions workflows for automated testing, linting, building, and quality checks for ComfyUI.

## Workflows Overview

### 1. Ruff Lint (`ruff-lint.yml`)
**Trigger**: Push/PR to master, main, develop branches

**Purpose**: Fast Python linting using Ruff

**Steps**:
- Syntax error detection (E9, F63, F7, F82)
- Full linting with all rules
- Code formatting check
- Statistics reporting

**Badge**: `[![Ruff](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/ruff-lint.yml/badge.svg)](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/ruff-lint.yml)`

### 2. Python Tests (`python-tests.yml`)
**Trigger**: Push/PR to master, main, develop branches + manual

**Purpose**: Run unit tests across multiple platforms

**Matrix**:
- OS: Ubuntu, Windows, macOS
- Python: 3.14

**Steps**:
- Install system dependencies
- Run pytest with coverage
- Upload coverage to Codecov
- Import validation

**Badge**: `[![Tests](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/python-tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/python-tests.yml)`

### 3. Build and Package (`build.yml`)
**Trigger**: Push to master/main, tags (v*), PR, manual

**Purpose**: Build distributable packages

**Steps**:
- Build Python packages (wheel, sdist)
- Upload artifacts
- Create GitHub releases (on tags)

**Badge**: `[![Build](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/build.yml/badge.svg)](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/build.yml)`

### 4. Code Quality (`code-quality.yml`)
**Trigger**: Push/PR + weekly schedule (Monday 00:00 UTC)

**Purpose**: Comprehensive code quality checks

**Jobs**:
- **Ruff Format**: Code formatting validation
- **Ruff Lint**: Linting with detailed reporting
- **Type Check**: Static type checking with mypy
- **Security Check**: Security vulnerability scan with Bandit
- **Dependency Check**: Check for vulnerable dependencies with pip-audit
- **Complexity Check**: Code complexity analysis with radon

**Badge**: `[![Code Quality](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/code-quality.yml/badge.svg)](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/code-quality.yml)`

### 5. CUDA Tests (`cuda-tests.yml`)
**Trigger**: Push/PR to master/main + manual

**Purpose**: CUDA compatibility testing

**Matrix**:
- CUDA versions: 12.8, 12.9

**Steps**:
- CUDA availability check
- PyTorch CUDA testing
- Numba CUDA testing
- CUDA 13.x configuration validation
- Python 3.14 compatibility check

**Badge**: `[![CUDA Tests](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/cuda-tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/cuda-tests.yml)`

### 6. Documentation (`documentation.yml`)
**Trigger**: Push/PR to master/main + manual

**Purpose**: Documentation validation and generation

**Jobs**:
- **Docs Check**: Verify required documentation files
- **Link Check**: Validate Markdown links
- **YAML Validation**: Ensure workflow files are valid
- **API Docs**: Generate API documentation with pdoc
- **Migration Guide**: Validate migration documentation completeness

**Badge**: `[![Docs](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/documentation.yml/badge.svg)](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/documentation.yml)`

## Ruff Configuration

Configuration file: `ruff.toml`

### Key Settings:
- **Target**: Python 3.14
- **Line Length**: 120 characters
- **Style**: Google docstring convention
- **Import Sorting**: isort integration

### Enabled Rules:
- E/W: pycodestyle
- F: Pyflakes
- I: isort
- N: pep8-naming
- UP: pyupgrade (Python 3.14 modernization)
- B: flake8-bugbear
- NPY: NumPy-specific rules
- Plus 20+ more rule sets

### Usage:

```bash
# Check for issues
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Format code
ruff format .

# Check formatting without changes
ruff format --check .
```

## Local Development

### Setup Pre-commit Hook

Create `.git/hooks/pre-commit`:
```bash
#!/bin/sh
echo "Running Ruff linter..."
ruff check . --select=E9,F63,F7,F82 --exit-non-zero-on-fix

if [ $? -ne 0 ]; then
    echo "Linting failed. Please fix errors before committing."
    exit 1
fi

echo "Running Ruff formatter check..."
ruff format --check .

if [ $? -ne 0 ]; then
    echo "Code formatting issues found. Run 'ruff format .' to fix."
    exit 1
fi

echo "All checks passed!"
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

### Run Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist pytest-timeout

# Run all tests
pytest tests-unit/ -v

# Run with coverage
pytest tests-unit/ --cov=. --cov-report=html

# Run specific test file
pytest tests-unit/comfy/test_model_management.py -v
```

### Lint Locally

```bash
# Install Ruff
pip install ruff

# Run linting
ruff check .

# Fix issues automatically
ruff check . --fix

# Format code
ruff format .
```

## Secrets Configuration

Required secrets in GitHub repository settings:

1. **GITHUB_TOKEN**: Automatically provided by GitHub
2. **CODECOV_TOKEN** (optional): For coverage reporting to Codecov

## Workflow Customization

### Changing Python Version

Edit matrix in workflow files:
```yaml
matrix:
  python-version: ['3.14', '3.15']  # Add more versions
```

### Adding New Platforms

Edit OS matrix:
```yaml
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest, macos-14]  # macos-14 is ARM
```

### Modifying Ruff Rules

Edit `ruff.toml`:
```toml
[tool.ruff.lint]
select = ["E", "F", "NEW_RULE"]
ignore = ["E501", "NEW_IGNORE"]
```

## Badge Placement

Add to main `README.md`:
```markdown
[![Ruff](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/ruff-lint.yml/badge.svg)](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/ruff-lint.yml)
[![Tests](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/python-tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/python-tests.yml)
[![Code Quality](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/code-quality.yml/badge.svg)](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/code-quality.yml)
[![CUDA](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/cuda-tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/ComfyUI/actions/workflows/cuda-tests.yml)
```

## Troubleshooting

### Workflow Not Triggering
- Check branch names match workflow triggers
- Verify `.github/workflows/` path is correct
- Check YAML syntax with `yamllint`

### Tests Failing
- Run tests locally first
- Check Python version compatibility
- Verify all dependencies installed

### Ruff Errors
- Run `ruff check .` locally to see errors
- Use `ruff check . --fix` for auto-fixes
- Check `ruff.toml` configuration

### CUDA Tests Skipped
- Normal if no GPU available
- GitHub runners don't have GPUs by default
- Use self-hosted runners with GPUs if needed

## Performance Optimization

### Caching
All workflows use pip caching:
```yaml
- uses: actions/setup-python@v5
  with:
    cache: 'pip'
```

### Parallel Execution
Tests use pytest-xdist for parallelization:
```bash
pytest -n auto  # Use all available cores
```

### Conditional Steps
Some steps skip based on conditions:
```yaml
if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.14'
```

## Best Practices

1. **Keep workflows fast**: Use caching, parallel execution
2. **Fail fast**: Use `fail-fast: false` for matrix builds
3. **Clear names**: Use descriptive job and step names
4. **Artifacts**: Upload important outputs for debugging
5. **Timeouts**: Set reasonable timeouts to prevent hung jobs
6. **Secrets**: Never commit secrets, use GitHub Secrets

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [pytest Documentation](https://docs.pytest.org/)
- [Codecov Documentation](https://docs.codecov.com/)

---

**Last Updated**: 2026-01-05
**Python Version**: 3.14
**Ruff Version**: Latest
**CUDA Support**: 13.x
