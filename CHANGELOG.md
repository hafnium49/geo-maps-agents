# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- PR #5: OR-Tools VRPTW sequencer
- PR #6: CI/CD & comprehensive test suite

## [0.4.0] - 2025-10-12

### Added - PR #4: HDBSCAN Fallback Logic
- **New Spatial Module**
  - `src/spatial/clustering.py` with robust clustering implementation (430+ lines)
  - `src/spatial/__init__.py` for clean public API exports
  - `ClusteringConfig` dataclass for configuration
  - `ClusteringDiagnostics` dataclass for quality metrics
  - `ClusterInfo` dataclass for cluster metadata
  
- **Fallback Logic**
  - `cluster_with_fallback()` main clustering function with comprehensive error handling
  - `_detect_degenerate_case()` for sparse data detection
  - `_handle_over_clustering()` with adaptive refitting
  - `_compute_cluster_quality()` with silhouette score calculation
  - `_fallback_to_scores()` for graceful degradation to score-only selection
  
- **Quality Assessment**
  - Silhouette score computation (range: -1 to 1)
  - Cluster size distribution tracking
  - Noise ratio monitoring
  - Actionable suggestions based on quality metrics
  
- **Enhanced Labeling**
  - `label_cluster()` with deterministic token selection
  - Generic token filtering ("point_of_interest", "establishment")
  - Top-N token aggregation (default: 2 tokens)
  - Graceful handling of empty/missing types
  
- **Diagnostics & Telemetry**
  - Comprehensive logging of clustering results
  - Fallback reason tracking for debugging
  - Refit attempt counting
  - Configuration tracking (actual config used after refitting)
  
- **Testing & Verification**
  - `verify_pr4.py` automated verification script (7 checks)
  - Degenerate case handling tests
  - Over-clustering detection tests
  - Successful clustering tests
  - Label determinism tests
  - Integration tests with geotrip_agent.py
  
- **Documentation**
  - `PR4_SUMMARY.md` comprehensive technical overview
  - `QUICK_REFERENCE_PR4.md` usage guide and API reference

### Changed
- **geotrip_agent.py**
  - `_hdbscan_clusters()` now returns `(hex_df, clusters, diagnostics)` tuple (was `(hex_df, clusters)`)
  - Enhanced with automatic diagnostics logging
  - Delegates to `src.spatial.cluster_with_fallback()`
  - `_label_cluster()` now delegates to `src.spatial.label_cluster()`
  - `spatial_context_and_scoring()` updated to handle diagnostics tuple
  
- **Clustering Algorithm**
  - Now detects degenerate cases (< min_cluster_size points)
  - Automatically refits when over-clustering detected (> max_clusters)
  - Computes silhouette scores for quality assessment
  - Falls back to score-only selection when clustering fails

### Deprecated
- Old `_hdbscan_clusters()` signature without diagnostics (backward compatible)
- Old `_label_cluster()` function (use `src.spatial.label_cluster()`)

### Improved
- **Robustness**
  - Never crashes on edge cases (too few points, HDBSCAN failures)
  - Graceful degradation to score-only selection
  - Adaptive refitting prevents over-clustering
  - Clear error messages with actionable suggestions
  
- **Quality Metrics**
  - Silhouette scores enable A/B testing measurement
  - Cluster size distribution helps identify fragmentation
  - Noise ratio tracking shows clustering effectiveness
  
- **Observability**
  - Every clustering run gets detailed diagnostics
  - Automatic logging of success/fallback status
  - Suggestions provided for poor-quality results
  - Refit attempts tracked for performance monitoring

- **Determinism**
  - Cluster labels are now deterministic (same input → same output)
  - Generic tokens filtered for more meaningful labels
  - Robust handling of tie-breaking in token selection

## [0.3.0] - 2025-10-12

### Added - PR #3: Scoring Normalization & A/B Harness
- **New Scoring Module**
  - `src/scoring/normalization.py` with percentile-based normalization (5th/95th)
  - `src/scoring/weights.py` with A/B testing support
  - `src/scoring/scorer.py` with telemetry logging
  - `src/scoring/__init__.py` for clean public API exports
  
- **Enhanced Normalization**
  - `percentile_norm()` for robust outlier handling
  - `normalize_eta()` with proper inversion (lower ETA = higher score)
  - `normalize_rating()`, `normalize_crowd_proxy()`, `normalize_diversity()`
  - Minimum ETA clamping (180s for foot/indoor routing)
  
- **A/B Testing Framework**
  - `select_ab_variant()` with session-sticky hash-based assignment
  - `VARIANT_A`, `VARIANT_B`, `VARIANT_C` predefined weight configurations
  - `WeightConfig` dataclass with variant names
  - `load_weights_from_yaml()` for configuration file support
  
- **Telemetry System**
  - `PlaceScorer` class with detailed per-stop logging
  - `ScoreBreakdown` dataclass with component scores
  - `ScoringTelemetry` dataclass with all scoring context
  - JSON export for analysis with `export_telemetry_json()`
  
- **Configuration**
  - `configs/weights.yaml` with 7 variant definitions:
    - `default`: Balanced configuration
    - `variant-a`: Quality-focused (+35% rating weight)
    - `variant-b`: Diversity-focused (+30% diversity weight)
    - `variant-c`: Proximity-focused (+25% ETA weight)
    - `variant-local`: Local experiences emphasis
    - `variant-fast`: Time-optimized
    - `variant-leisurely`: Quality over quantity
  
- **Testing & Verification**
  - `verify_pr3.py` automated verification script (15 checks)
  - Comprehensive functional tests for normalization, A/B testing, telemetry

### Changed
- **geotrip_agent.py**
  - `_score_places()` now delegates to `src.scoring.PlaceScorer`
  - Automatic telemetry logging with top-3 score summaries
  - `WeightConfig` now includes `variant_name` for tracking
  - `_robust_norm()` now delegates to `percentile_norm()`
  
- **Scoring Algorithm**
  - ETA now properly inverted (lower travel time = higher score)
  - Percentile-based normalization replaces min/max (handles outliers)
  - Crowd penalty properly applied as subtraction
  - Score breakdown logged for debugging and A/B measurement

### Deprecated
- `_robust_norm()` function (use `src.scoring.percentile_norm()`)
- Old min/max normalization approach

### Improved
- **Scoring Accuracy**
  - Percentile normalization prevents outlier distortion
  - ETA properly inverted: 300s scores higher than 900s
  - More stable scores across different datasets
  
- **A/B Testing**
  - Deterministic variant assignment (same user → same variant)
  - SHA256-based hashing prevents bias
  - Supports user_id, device_id, session_id identifiers
  
- **Observability**
  - Every place gets detailed telemetry
  - Score breakdown shows contribution of each component
  - Variant name tracked for measurement
  - Raw values preserved for debugging

## [0.2.0] - 2025-10-12

### Added - PR #2: Matrix Guardrails & Caching
- **New Routing Module**
  - `src/routing/matrix.py` with comprehensive matrix computation logic (450+ lines)
  - `src/routing/__init__.py` for clean public API exports
  - `MatrixCache` class with dual-TTL caching (5min traffic / 60min static)
  - `TravelMode` and `RoutingPreference` enums for type safety
  - `MatrixRequest` dataclass for structured requests
  
- **Enhanced Error Handling**
  - `validate_matrix_request()` with detailed error messages
  - Helpful suggestions when limits exceeded (5 actionable fixes)
  - Dynamic limit calculation with `get_matrix_limits()`
  
- **Improved Retry Logic**
  - `exponential_backoff_with_jitter()` for better retry behavior
  - Truncated exponential growth with randomization
  - Configurable max backoff time (8 seconds)
  
- **Future-Ready Features**
  - `compute_route_matrix_streaming()` placeholder for gRPC
  - Cache statistics with `get_cache_stats()`
  - Cache management with `clear_cache()`
  
- **Testing & Verification**
  - `verify_pr2.py` automated verification script (10 checks)
  - `PR2_SUMMARY.md` comprehensive PR documentation

### Changed
- **geotrip_agent.py**
  - `route_matrix()` now uses enhanced `src.routing` module
  - Automatic conversion from strings to type-safe enums
  - Delegates to `compute_route_matrix()` with full guardrails
  
### Deprecated
- `_matrix_limit()` function (use `src.routing.get_matrix_limits()`)
- `_backoff_sleep()` function (use `src.routing.matrix.exponential_backoff_with_jitter()`)
- Direct `_matrix_cache` access (use `src.routing` cache system)

### Improved
- **Cache Efficiency**
  - Static routes cached 12× longer (60min vs 5min)
  - Separate caches prevent traffic routes from evicting static ones
  - Automatic cache selection based on routing preference
  
- **Error Messages**
  - Now show requested vs maximum elements
  - Provide 5 specific suggestions to fix limit violations
  - Include mode-specific guidance

- **Type Safety**
  - Enums prevent invalid mode/preference values
  - Dataclasses ensure structured data
  - Better IDE autocomplete and type checking

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
