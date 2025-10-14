# Quick Development Commands

## Initial Setup (One-Time)

```bash
# 1. Create virtual environment
uv venv

# 2. Install all dependencies
uv pip install -e ".[dev]"

# 3. Set up environment variables
cp .env.sample .env
# Edit .env and add your API keys

# 4. Verify installation
uv run python docs/prs/verify_pr2.py
```

## Daily Development

### Running Scripts
```bash
uv run python docs/prs/verify_pr2.py   # Verify PR #2 implementation
uv run geotrip_agent.py                 # Run main agent
uv run python -m src.routing            # Run specific module
```

### Testing
```bash
uv run pytest                                    # Run all tests
uv run pytest tests/test_routing.py             # Run specific test file
uv run pytest -v                                 # Verbose output
uv run pytest --cov=src --cov-report=html       # With coverage
uv run pytest -k "test_matrix"                  # Run tests matching pattern
```

### Code Quality
```bash
uv run black .                                   # Format all code
uv run black src/ geotrip_agent.py              # Format specific files
uv run ruff check .                             # Lint all code
uv run ruff check --fix .                       # Lint and auto-fix
uv run mypy src/                                # Type check
```

### Package Management
```bash
uv pip install package-name                     # Add new package
uv pip uninstall package-name                   # Remove package
uv pip list                                      # List installed packages
uv pip freeze                                    # Show exact versions
```

### Verification
```bash
# Quick verification of PR #2
uv run python docs/prs/verify_pr2.py

# Expected output:
# 🎉 All checks passed! (10/10)
# ✅ PR #2 is complete and working correctly!
```

## Environment Management

```bash
# Create new virtual environment
uv venv

# Activate virtual environment (optional with uv run)
source .venv/bin/activate

# Deactivate
deactivate

# Recreate environment from scratch
rm -rf .venv && uv venv && uv pip install -e ".[dev]"
```

## Common Tasks

### Add a New Dependency
```bash
# 1. Install it
uv pip install new-package

# 2. Add to pyproject.toml dependencies
# Edit pyproject.toml manually

# 3. Reinstall to verify
uv pip install -e .
```

### Run PR Verification
```bash
uv run python docs/prs/verify_pr2.py
# Should show: 🎉 All checks passed! (10/10)
```

### Check Code Quality Before Committing
```bash
uv run black .
uv run ruff check --fix .
uv run mypy src/
uv run pytest
```

### Debug Import Issues
```bash
# Verify package is installed in editable mode
uv pip list | grep geo-maps-agents

# Should show: geo-maps-agents 0.1.0 (from file:///path/to/geo-maps-agents)

# Reinstall if needed
uv pip install -e .
```

## Project-Specific Commands

### Check Configuration
```bash
# View city configs
cat configs/dense-city.yaml
cat configs/suburban.yaml
cat configs/rural.yaml

# Check environment
cat .env
```

### View API Field Masks
```bash
uv run python -c "from src.tools.fields import *; print(PLACES_TEXT_SEARCH_FIELDS)"
```

### Test Matrix Limits
```bash
uv run python -c "
from src.routing import get_matrix_limits, TravelMode, RoutingPreference
limits = get_matrix_limits(TravelMode.TRANSIT, RoutingPreference.TRAFFIC_AWARE)
print(f'Max elements: {limits.max_elements}')
"
```

### Test Caching
```bash
uv run python -c "
from src.routing import get_cache_stats
stats = get_cache_stats()
print('Cache stats:', stats)
"
```

## Troubleshooting

### "Module not found" errors
```bash
uv pip install -e .
```

### "Missing GOOGLE_MAPS_API_KEY" error
```bash
cp .env.sample .env
# Edit .env and add your keys
```

### Dependency conflicts
```bash
rm -rf .venv
uv venv
uv pip install -e ".[dev]"
```

### Tests failing
```bash
# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_routing.py::test_matrix_limits -v
```

## Help

```bash
uv --help                       # uv commands
uv pip --help                   # pip commands
uv run pytest --help            # pytest options
uv run black --help             # black options
uv run ruff --help              # ruff options
```

## Status Check

```bash
# Quick status check of your development environment
echo "Python version:" && python --version
echo "uv version:" && uv --version
echo "Installed packages:" && uv pip list | wc -l
echo "Virtual env:" && which python
echo "Environment vars:" && grep -c "KEY" .env 2>/dev/null || echo "No .env file"
```
