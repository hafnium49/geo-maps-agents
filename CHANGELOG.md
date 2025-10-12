# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- PR #2: Matrix guardrails & caching improvements
- PR #3: A/B testing harness & scoring normalization
- PR #4: HDBSCAN fallback logic
- PR #5: OR-Tools VRPTW sequencer
- PR #6: CI/CD & comprehensive test suite

## [0.1.0] - 2025-10-12

### Added - PR #1: Config & Secrets Infrastructure
- **Dependency Management**
  - `pyproject.toml` with pinned dependencies for reproducible builds
  - `requirements.txt` for traditional pip installation
  - `requirements-dev.txt` for development dependencies
  
- **Environment Configuration**
  - `.env.sample` template for API keys and configuration
  - `python-dotenv` integration for automatic environment loading
  - Enhanced error messages for missing API keys
  
- **Centralized FieldMasks**
  - `src/tools/fields.py` with all Google Maps API field definitions
  - Helper functions: `get_places_search_mask()`, `get_places_details_mask()`, `get_routes_matrix_mask()`
  - Comprehensive documentation for each field's purpose
  
- **City Profile Configurations**
  - `configs/dense-city.yaml` for high-density urban environments
  - `configs/suburban.yaml` for medium-density suburban areas
  - `configs/rural.yaml` for low-density rural regions
  - `src/tools/config_loader.py` for YAML configuration loading
  
- **Documentation**
  - Complete README rewrite with:
    - Table of contents and navigation
    - Improved quickstart guide
    - Configuration documentation
    - **ToS Guardrails section** (Google Maps Platform compliance)
    - Architecture overview
    - Development guide
  - `PR1_SUMMARY.md` detailed PR documentation
  - `verify_pr1.py` automated verification script
  
- **Project Structure**
  - `src/` source package directory
  - `src/tools/` for utilities and API clients
  - Proper `__init__.py` files for clean imports
  - Comprehensive `.gitignore` rules

### Changed
- **geotrip_agent.py**
  - Now imports centralized FieldMasks instead of hardcoded field lists
  - Uses `get_places_search_mask()` in `places_text_search()`
  - Uses `get_places_details_mask()` in `place_details()`
  - Uses `get_routes_matrix_mask()` in `route_matrix()`
  - Enhanced error message for missing `GOOGLE_MAPS_API_KEY`
  - Added deprecation note for legacy `_fieldmask()` helper

### Deprecated
- Inline FieldMask definitions (replaced by centralized `src/tools/fields.py`)
- Direct environment variable access without validation

## [0.0.1] - 2025-10-11 (Pre-PR baseline)

### Initial Implementation
- Basic OpenAI Agents SDK integration
- Google Places API Text Search and Details
- Google Routes API matrix computation
- H3 spatial indexing (resolution 9)
- HDBSCAN clustering
- Greedy TSP-TW sequencing
- deck.gl visualization on Google Maps
- Multi-agent orchestration (data, spatial, UX agents)

---

## Version History

- **0.1.0** (Current): Config & secrets infrastructure ← PR #1 complete
- **0.0.1**: Initial proof-of-concept

## Upgrade Guide

### From 0.0.1 to 0.1.0

1. **Install new dependencies:**
   ```bash
   pip install -e .
   # or
   pip install -r requirements.txt
   ```

2. **Create environment file:**
   ```bash
   cp .env.sample .env
   # Edit .env and add your API keys
   ```

3. **Update imports (if you're importing from geotrip_agent.py):**
   ```python
   # Old (0.0.1):
   from geotrip_agent import places_text_search
   
   # New (0.1.0):
   # Use the script directly, or import from src.tools for utilities
   from src.tools.fields import get_places_search_mask
   ```

4. **Verify installation:**
   ```bash
   python verify_pr1.py
   ```

## Contribution Guidelines

When making changes:
1. Update this CHANGELOG under `[Unreleased]`
2. Follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format
3. Use sections: Added, Changed, Deprecated, Removed, Fixed, Security
4. Include PR number references when applicable

---

**Legend:**
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes
