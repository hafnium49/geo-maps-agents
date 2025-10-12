# Development Environment Setup

This guide shows how to set up a development environment for the geo-maps-agents project using `uv`.

## Prerequisites

- Python 3.10 or higher
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))

## Quick Setup

### 1. Create Virtual Environment

```bash
uv venv
```

This creates a `.venv` directory with an isolated Python environment.

### 2. Install Dependencies

```bash
# Install project in editable mode with all dependencies
uv pip install -e .

# Install dev dependencies (testing, linting)
uv pip install -e ".[dev]"
```

### 3. Set Up Environment Variables

```bash
# Copy the sample environment file
cp .env.sample .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY from https://platform.openai.com/api-keys
# - GOOGLE_MAPS_API_KEY from https://console.cloud.google.com/google/maps-apis/credentials
```

### 4. Verify Installation

```bash
# Run PR #2 verification tests
uv run verify_pr2.py

# You should see: 🎉 All checks passed! (10/10)
```

## Using uv for Development

### Run Scripts

```bash
# Run any Python script with uv
uv run verify_pr2.py
uv run geotrip_agent.py

# Run pytest
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Format and Lint Code

```bash
# Format with black
uv run black src/ geotrip_agent.py

# Lint with ruff
uv run ruff check src/ geotrip_agent.py

# Type check with mypy
uv run mypy src/
```

### Install Additional Packages

```bash
# Install a new package
uv pip install package-name

# Update pyproject.toml dependencies manually, then:
uv pip sync
```

## Project Structure

```
geo-maps-agents/
├── .venv/                      # Virtual environment (created by uv)
├── src/
│   ├── routing/                # PR #2: Matrix computation module
│   │   ├── __init__.py
│   │   └── matrix.py
│   ├── tools/                  # PR #1: Utilities and helpers
│   │   ├── config_loader.py
│   │   └── fields.py
│   └── (future modules)
├── configs/                    # City profile configurations
│   ├── dense-city.yaml
│   ├── suburban.yaml
│   └── rural.yaml
├── geotrip_agent.py           # Main agent orchestration
├── pyproject.toml             # Project configuration
├── .env                       # API keys (git-ignored)
└── .env.sample                # Template for .env
```

## Development Workflow

### 1. Activate Virtual Environment (Optional)

```bash
source .venv/bin/activate
```

Note: With `uv run`, you don't need to activate the venv manually.

### 2. Make Changes

Edit code in `src/` or `geotrip_agent.py`.

### 3. Run Tests

```bash
uv run verify_pr2.py
# or
uv run pytest
```

### 4. Format Code

```bash
uv run black .
uv run ruff check --fix .
```

### 5. Verify Everything Works

```bash
# Run all checks
uv run verify_pr2.py

# Run specific tests
uv run pytest tests/test_routing.py -v
```

## Troubleshooting

### Issue: Import errors

**Solution**: Make sure you installed the package in editable mode:
```bash
uv pip install -e .
```

### Issue: Missing API keys

**Solution**: Create `.env` file with valid API keys:
```bash
cp .env.sample .env
# Edit .env and add your keys
```

### Issue: Dependency conflicts

**Solution**: Recreate the virtual environment:
```bash
rm -rf .venv
uv venv
uv pip install -e ".[dev]"
```

### Issue: Module not found errors

**Solution**: Ensure you're running scripts from the project root:
```bash
cd /path/to/geo-maps-agents
uv run verify_pr2.py
```

## Dependencies

### Core Dependencies
- `openai-agents>=0.3.3` - Agent orchestration framework
- `httpx>=0.27.0` - Async HTTP client
- `h3>=3.7.6,<4.0.0` - Hexagonal spatial indexing
- `hdbscan>=0.8.33` - Density-based clustering
- `ortools>=9.10.0` - Optimization tools
- `pydantic>=2.0.0` - Data validation
- `cachetools>=5.3.0` - Caching utilities
- `python-dotenv>=1.0.0` - Environment variable management

### Dev Dependencies
- `pytest>=8.0.0` - Testing framework
- `pytest-cov>=5.0.0` - Coverage reporting
- `pytest-asyncio>=0.23.0` - Async test support
- `ruff>=0.5.0` - Fast Python linter
- `mypy>=1.10.0` - Static type checker
- `black>=24.0.0` - Code formatter

## Next Steps

After setting up your development environment:

1. ✅ PR #1: Config & Secrets Infrastructure (Complete)
2. ✅ PR #2: Matrix Guardrails & Caching (Complete)
3. 🔜 PR #3: Scoring Normalization & A/B Harness
4. 🔜 PR #4: HDBSCAN Fallback Logic
5. 🔜 PR #5: OR-Tools VRPTW Sequencer
6. 🔜 PR #6: CI & Testing Infrastructure

Run `uv run verify_pr2.py` to verify your setup is working correctly!

## Additional Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- [Google Maps Platform](https://developers.google.com/maps)
- [H3 Spatial Index](https://h3geo.org/)
- [OR-Tools](https://developers.google.com/optimization)
