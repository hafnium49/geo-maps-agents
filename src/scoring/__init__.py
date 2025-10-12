"""
Scoring Module for GeoTrip Agent

Provides comprehensive scoring functionality with:
- Percentile-based normalization (5th/95th)
- A/B testing with session-sticky variant selection
- Detailed per-stop telemetry logging
- Weight management and configuration

Usage:
    from src.scoring import (
        score_places,
        PlaceScorer,
        WeightConfig,
        select_ab_variant,
        normalize_rating,
        normalize_eta,
    )
    
    # Simple scoring
    scored_df = score_places(places_df, etas, hex_df, weights=my_weights)
    
    # A/B testing
    weights = select_ab_variant(user_id="user_123")
    scorer = PlaceScorer(weights=weights, enable_telemetry=True)
    scored_df = scorer.score_places(places_df, etas, hex_df)
    telemetry = scorer.get_telemetry()
"""

# Normalization functions
from .normalization import (
    percentile_norm,
    robust_norm,
    normalize_rating,
    normalize_eta,
    normalize_crowd_proxy,
    normalize_diversity,
    normalize_localness,
)

# Weight management and A/B testing
from .weights import (
    WeightConfig,
    DEFAULT_WEIGHTS,
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    WEIGHT_VARIANTS,
    select_ab_variant,
    get_variant_by_name,
    load_weights_from_yaml,
    save_weights_to_yaml,
)

# Scoring engine
from .scorer import (
    PlaceScorer,
    ScoreBreakdown,
    ScoringTelemetry,
    score_places,
)

__all__ = [
    # Normalization
    "percentile_norm",
    "robust_norm",
    "normalize_rating",
    "normalize_eta",
    "normalize_crowd_proxy",
    "normalize_diversity",
    "normalize_localness",
    
    # Weights
    "WeightConfig",
    "DEFAULT_WEIGHTS",
    "VARIANT_A",
    "VARIANT_B",
    "VARIANT_C",
    "WEIGHT_VARIANTS",
    "select_ab_variant",
    "get_variant_by_name",
    "load_weights_from_yaml",
    "save_weights_to_yaml",
    
    # Scoring
    "PlaceScorer",
    "ScoreBreakdown",
    "ScoringTelemetry",
    "score_places",
]
