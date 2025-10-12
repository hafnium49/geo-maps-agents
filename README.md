# geo-maps-agents

This repository hosts a production-flavored OpenAI Agents SDK reference implementation for orchestrating multi-stage geospatial trip planning. The core entry point is [`geotrip_agent.py`](geotrip_agent.py), which wires together:

- A triage/orchestrator agent coordinating dedicated data, spatial, and UX agents.
- Google Places Text Search and Details (New) calls with FieldMasks for cost-aware discovery.
- Traffic-aware travel time matrices, H3 binning, HDBSCAN clustering, and heuristic TSP-TW sequencing.
- A deck.gl overlay rendered on Google Maps JS to remain Terms-of-Service compliant.

## 📋 Table of Contents

- [Quickstart](#quickstart)
- [Configuration](#configuration)
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

### 3. Run Example

```bash
# Run the Tokyo Station example (13:00–18:00, transit mode)
python geotrip_agent.py
```

**Expected Output:**
- `geotrip_map.html` created with Google Map + deck.gl overlay
- Console shows: "N stops scheduled; M clusters found"
- Per-stop details with score breakdown and reasoning

### 4. Open Map Visualization

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
├── .env.sample               # Environment template
├── configs/                  # City profiles
│   ├── dense-city.yaml
│   ├── suburban.yaml
│   └── rural.yaml
├── src/
│   └── tools/
│       ├── fields.py         # FieldMask constants
│       └── config_loader.py  # YAML loader
└── README.md
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (coming in PR #6)
pytest

# Run linter
ruff check .

# Format code
black .
```

### Contributing

See upcoming PRs:
- **PR #2**: Matrix guardrails & caching improvements
- **PR #3**: A/B testing harness & scoring normalization
- **PR #4**: HDBSCAN fallback logic
- **PR #5**: OR-Tools VRPTW sequencer
- **PR #6**: CI/CD & comprehensive test suite

## 📝 License

MIT

## 🙏 Acknowledgments

- OpenAI Agents SDK
- Google Maps Platform
- H3 (Uber's Hexagonal Hierarchical Geospatial Indexing System)
- deck.gl (Uber's WebGL visualization framework)