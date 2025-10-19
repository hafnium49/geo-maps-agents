# geo-maps-agents

This repository now ships two complementary experiences for geospatial trip planning:

- A production-flavored OpenAI **Apps SDK (ChatGPT App)** built on a FastAPI MCP server (`apps/mcp_server`) with deck.gl + Google Maps widgets compiled from `apps/widgets`.
- The original OpenAI **Agents SDK** reference implementation (`geotrip_agent.py`) which demonstrates multi-agent orchestration.

The MCP server exposes three actions (`search_places`, `optimize_itinerary`, `details`) that the ChatGPT UI can call directly. It reuses the battle-tested scoring, routing, and clustering modules and emits structured JSON plus widget metadata so the ChatGPT App can render an interactive itinerary composed of:

- Google Maps-compliant deck.gl overlays for H3 heatmaps, clustered POIs, isochrone rings, and animated route playback.
- A rich day timeline widget that mirrors the sequencing output from the VRPTW solver.

The legacy Agents SDK entry point [`geotrip_agent.py`](geotrip_agent.py) continues to orchestrate:

- A triage/orchestrator agent coordinating dedicated data, spatial, and UX agents.
- Google Places Text Search and Details (New) calls with FieldMasks for cost-aware discovery.
- Traffic-aware travel time matrices, H3 binning, HDBSCAN clustering, and heuristic TSP-TW sequencing.
- A deck.gl overlay rendered on Google Maps JS to remain Terms-of-Service compliant.

## 📋 Table of Contents

- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Testing](#testing)
- [ToS Guardrails](#tos-guardrails)
- [Architecture](#architecture)
- [API Requirements](#api-requirements)
- [Development](#development)

## 🚀 Quickstart

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the sample environment file
cp .env.sample .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY: Get from https://platform.openai.com/api-keys
# - GOOGLE_MAPS_API_KEY: Get from https://console.cloud.google.com/google/maps-apis/credentials
```

**Required Google APIs:**
- Places API (New)
- Routes API

### 3. Run the ChatGPT App MCP Server

```bash
# Build the visualization widgets (only when assets change)
cd apps/widgets
pnpm install  # or npm install / yarn install
pnpm build    # outputs assets to ../mcp_server/assets/

# Launch the MCP server
cd ../..
uvicorn apps.mcp_server.main:app --host 0.0.0.0 --port 5050
```

In ChatGPT, add a custom app that points to `http://localhost:5050`, upload `app.manifest.json`, and test the `search_places`, `optimize_itinerary`, and `details` actions. The ChatGPT UI will stream the returned widgets from `/assets`.

### 4. Run the legacy Agents SDK example

```bash
# Run the Tokyo Station example (13:00–18:00, transit mode)
python geotrip_agent.py
```

**Expected Output:**
- `geotrip_map.html` created with Google Map + deck.gl overlay
- Console shows: "N stops scheduled; M clusters found"
- Per-stop details with score breakdown and reasoning

### 5. Open Map Visualization

```bash
# Open the generated map in your browser
open geotrip_map.html  # macOS
xdg-open geotrip_map.html  # Linux
start geotrip_map.html  # Windows
```

## ⚙️ Configuration

### City Profiles

Pre-configured profiles for different urban densities:

- **`dense-city.yaml`**: High POI density (Tokyo, NYC, London)
  - H3 resolution 9 (~175m hexagons)
  - Transit-first routing
  - 12+ POIs per cluster

- **`suburban.yaml`**: Medium density (Silicon Valley, outer boroughs)
  - H3 resolution 8 (~461m hexagons)
  - Car-dependent routing
  - 8+ POIs per cluster

- **`rural.yaml`**: Low density (national parks, countryside)
  - H3 resolution 7 (~1.22km hexagons)
  - Drive-only routing
  - 5+ POIs per cluster

**Usage:**
```bash
# Set via environment variable
export CITY_PROFILE=suburban

# Or specify in .env file
echo "CITY_PROFILE=suburban" >> .env
```

## 🧪 Testing

### Quick Start

```bash
# Install test dependencies
uv sync  # Or: pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Suite Overview

**Total: 104 tests** (all passing ✅)

#### Unit Tests (86 tests)

- **Scoring Module** (`tests/test_scoring.py` - 30 tests)
  - Percentile-based normalization (5th/95th)
  - Weight configuration & A/B testing
  - PlaceScorer with telemetry
  - Edge cases (empty data, missing ratings)

- **Spatial Module** (`tests/test_spatial.py` - 28 tests)
  - H3 aggregation & neighbor detection
  - HDBSCAN clustering with fallback
  - Cluster labeling & quality metrics
  - Degenerate case handling

- **Routing Module** (`tests/test_routing.py` - 28 tests)
  - Distance matrix validation & caching
  - VRPTW solver with OR-Tools
  - Greedy sequencing fallback
  - Time window constraints

#### Integration Tests (18 tests)

- **End-to-End Pipeline** (`tests/test_integration.py`)
  - Complete flow: places → scoring → clustering → sequencing
  - Error handling & graceful degradation
  - Fallback mechanisms (clustering, VRPTW → greedy)
  - A/B testing variants
  - Output format validation
  - Performance benchmarks

### Test Categories

```bash
# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Run slow tests (performance benchmarks)
uv run pytest -m slow

# Run specific test file
uv run pytest tests/test_scoring.py -v
```

### Mock Infrastructure

All tests use **mock API clients** to avoid real API calls:

- **Mock Places API**: Returns fixtures from `tests/fixtures/places_api.json`
- **Mock Routes API**: Returns fixtures from `tests/fixtures/routes_api.json`
- **No network calls**: Tests run fast (~25 seconds for full suite)
- **No API costs**: Zero charges during testing

### CI/CD Pipeline

[![CI/CD](https://github.com/hafnium49/geo-maps-agents/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/hafnium49/geo-maps-agents/actions)

Automated testing runs on every push and pull request:

- ✅ **Lint**: Ruff code quality checks
- ✅ **Test**: Full test suite on Python 3.11 & 3.12
- ✅ **Coverage**: Must maintain ≥80% code coverage
- ✅ **Report**: Coverage uploaded to Codecov

**Workflow:** `.github/workflows/ci.yml`

### Coverage Requirements

- **Target**: ≥80% code coverage
- **Current**: (measured by CI)
- **Reports**: HTML coverage report available as CI artifact

### Writing Tests

```python
# Example unit test
import pytest
from src.scoring import PlaceScorer, WeightConfig

def test_place_scorer():
    weights = WeightConfig(w_rating=0.5, w_eta=0.3)
    scorer = PlaceScorer(weights=weights)
    # ... test implementation

# Example integration test
@pytest.mark.integration
def test_complete_pipeline(sample_places, mock_places_api):
    # Test end-to-end flow with mocks
    pass
```

For detailed testing documentation, see [`TESTING.md`](TESTING.md) (optional).

### FieldMask Centralization

All Google Maps API field selections are centralized in [`src/tools/fields.py`](src/tools/fields.py):

```python
from src.tools.fields import (
    get_places_search_mask,    # Text/Nearby Search
    get_places_details_mask,   # Place Details
    get_routes_matrix_mask,    # Route Matrix
)
```

This ensures:
- ✅ Consistent field selection across all API calls
- ✅ Minimized API costs (request only necessary data)
- ✅ Single source of truth for field definitions

## 🛡️ ToS Guardrails

### Google Maps Platform Terms of Service

This implementation strictly adheres to [Google Maps Platform ToS](https://cloud.google.com/maps-platform/terms):

#### ✅ Required Compliance

1. **Places Data Display**
   - ✅ All Places data **must be displayed on a Google Map**
   - ✅ We use Google Maps JavaScript API with deck.gl overlay
   - ❌ **Never** mix Places content with OpenStreetMap, Mapbox, or MapLibre

2. **Attribution Requirements**
   - ✅ Google Maps attribution automatically preserved
   - ✅ `googleMapsUri` deep-links maintained for each place
   - ✅ Places API attribution shown in UI

3. **Review Handling**
   - ✅ Display maximum 5 reviews per place
   - ✅ Include review author attribution
   - ❌ **Never cache reviews** - fetch fresh on each request
   - ✅ Show reviews with proper timestamp and rating

4. **Data Caching Policy**
   - ✅ Cache only Place IDs (not full place data)
   - ✅ Route matrices cached with TTL:
     - 5 minutes for traffic-aware routes
     - 60 minutes for non-traffic routes
   - ❌ **Never** store or cache review content

#### 🚫 Prohibited Actions

The input guardrail (`tos_guardrail`) actively blocks:
- Using Places data on non-Google maps
- Mixing data sources (OSM + Google Places)
- Requests that violate attribution requirements

```python
# Guardrail automatically rejects prompts like:
"Show this on OpenStreetMap"  # ❌ Blocked
"Mix with Mapbox"              # ❌ Blocked
"Cache reviews for later"      # ❌ Blocked
```

## 🏗️ Architecture

### Agent Handoff Flow

```
User Request
    ↓
┌─────────────────────────────┐
│  GeoTrip Orchestrator       │
│  (Triage Agent)             │
└─────────────────────────────┘
    ↓
    ├→ Data Agent (Places Search + Details)
    │     → Discovers POIs with FieldMasks
    │     → Enriches top candidates
    │
    ├→ Spatial Agent (Routes + Scoring)
    │     → Computes travel time matrix
    │     → H3 spatial indexing
    │     → HDBSCAN clustering
    │     → Multi-factor scoring
    │
    └→ UX Agent (Visualization)
          → Generates deck.gl overlay
          → Writes geotrip_map.html
          → Returns structured output
```

### Key Components

- **FieldMasks**: [`src/tools/fields.py`](src/tools/fields.py)
- **Config Loader**: [`src/tools/config_loader.py`](src/tools/config_loader.py)
- **City Profiles**: [`configs/`](configs/)
- **Main Agent**: [`geotrip_agent.py`](geotrip_agent.py)

## 🔑 API Requirements

### Google Maps Platform

Enable these APIs in [Google Cloud Console](https://console.cloud.google.com/google/maps-apis/):

1. **Places API (New)**
   - Text Search
   - Place Details
   - Nearby Search

2. **Routes API**
   - Compute Routes
   - Compute Route Matrix

**Cost Optimization:**
- FieldMasks limit data fetched (lower SKU pricing)
- Text Search (0.032 USD per request) + Details as needed
- Route Matrix cached with appropriate TTL

### OpenAI API

- GPT-4.1 for orchestrator agent
- GPT-4.1-mini for data/spatial/UX agents
- Function calling enabled

## 🔧 Development

### Project Structure

```
geo-maps-agents/
├── geotrip_agent.py          # Main entry point
├── pyproject.toml            # Dependencies & config
├── pytest.ini                # Test configuration
├── .coveragerc               # Coverage settings
├── .env.sample               # Environment template
├── configs/                  # City profiles
│   ├── dense-city.yaml
│   ├── suburban.yaml
│   └── rural.yaml
├── src/
│   ├── scoring/              # Scoring & normalization
│   ├── spatial/              # H3 & clustering
│   ├── routing/              # Matrix, VRPTW, greedy
│   └── tools/
│       ├── fields.py         # FieldMask constants
│       └── config_loader.py  # YAML loader
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── fixtures/             # Mock API responses
│   ├── test_scoring.py       # 30 unit tests
│   ├── test_spatial.py       # 28 unit tests
│   ├── test_routing.py       # 28 unit tests
│   └── test_integration.py   # 18 integration tests
└── README.md
```

### Running Tests

```bash
# Quick test run
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=term-missing

# Specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only

# Run with verbose output
uv run pytest -v

# Run specific file
uv run pytest tests/test_scoring.py
```

### Code Quality

```bash
# Run linter
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Type checking (if configured)
uv run mypy src/
```

### CI/CD

All tests run automatically on GitHub Actions:
- Lint checks (ruff)
- Unit + integration tests
- Coverage reporting (≥80% required)
- Python 3.11 & 3.12 compatibility
```

### Contributing

**Completed PRs:**
- ✅ **PR #1**: Config & Secrets Management
- ✅ **PR #2**: Matrix Guardrails & Caching
- ✅ **PR #3**: Scoring Normalization & A/B Testing
- ✅ **PR #4**: HDBSCAN Fallback Logic
- ✅ **PR #5**: OR-Tools VRPTW Sequencer
- ✅ **PR #6**: CI/CD & Testing Infrastructure (104 tests)

## 📝 License

MIT

## 🙏 Acknowledgments

- OpenAI Agents SDK
- Google Maps Platform
- H3 (Uber's Hexagonal Hierarchical Geospatial Indexing System)
- deck.gl (Uber's WebGL visualization framework)