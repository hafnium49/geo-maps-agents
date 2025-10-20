"""
Place Scoring Module with Telemetry

Provides comprehensive scoring logic with per-stop telemetry logging
for A/B testing measurement and debugging.
"""

from typing import Dict, List, Optional, Any, Mapping
from dataclasses import dataclass, asdict
import logging
import json
import math

import numpy as np
import pandas as pd
import h3

from .normalization import (
    normalize_rating,
    normalize_eta,
    normalize_crowd_proxy,
    normalize_diversity,
    normalize_localness,
)
from .weights import WeightConfig, DEFAULT_WEIGHTS


HEX_SCORE_DEFAULT_WEIGHTS: Dict[str, float] = {
    "localness": 0.35,
    "poi_density": 0.25,
    "open_coverage": 0.20,
    "reviews_sum": 0.20,
}


def _normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    min_val = series.min()
    max_val = series.max()
    if math.isclose(max_val, min_val):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


def score_hexes(
    hex_df: pd.DataFrame,
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Return ``hex_df`` with normalised metrics and composite score."""

    if hex_df.empty:
        return hex_df

    data = hex_df.copy()
    weight_map = {**HEX_SCORE_DEFAULT_WEIGHTS, **(dict(weights or {}))}

    data["density_norm"] = _normalize(data["poi_density"].astype(float))
    data["reviews_norm"] = _normalize(np.log1p(data["reviews_sum"].clip(lower=0.0)))
    data["open_coverage_norm"] = data["open_coverage"].clip(lower=0.0, upper=1.0)
    data["localness_norm"] = _normalize(data["localness"].astype(float))

    score = (
        weight_map["localness"] * data["localness_norm"]
        + weight_map["poi_density"] * data["density_norm"]
        + weight_map["open_coverage"] * data["open_coverage_norm"]
        + weight_map["reviews_sum"] * data["reviews_norm"]
    )

    max_weight = sum(weight_map.values()) or 1.0
    data["hex_score"] = score / max_weight
    return data


# Set up logging for telemetry
logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """
    Detailed breakdown of score components for telemetry.
    
    Attributes:
        rating_score: Contribution from rating (normalized * weight)
        diversity_score: Contribution from diversity gain
        eta_score: Contribution from travel time (inverted)
        open_score: Contribution from open-now status
        crowd_score: Contribution from crowd proxy (penalty)
        preference_multiplier: User preference adjustment
        final_score: Total weighted score
    """
    rating_score: float
    diversity_score: float
    eta_score: float
    open_score: float
    crowd_score: float
    preference_multiplier: float
    final_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ScoringTelemetry:
    """
    Per-stop telemetry data for A/B testing and debugging.
    
    Attributes:
        place_id: Unique identifier for the place
        place_name: Human-readable place name
        variant_name: A/B test variant used for scoring
        weights: Weight configuration used
        breakdown: Detailed score breakdown
        eta_sec: Travel time in seconds
        is_open_now: Whether place is currently open
        cluster_id: Cluster assignment (if any)
        cluster_label: Human-readable cluster label
        raw_values: Raw unnormalized values for debugging
    """
    place_id: str
    place_name: str
    variant_name: str
    weights: Dict[str, float]
    breakdown: ScoreBreakdown
    eta_sec: int
    is_open_now: bool
    cluster_id: Optional[int]
    cluster_label: Optional[str]
    raw_values: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["breakdown"] = self.breakdown.to_dict()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string for logging."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class PlaceScorer:
    """
    Main scoring engine with telemetry support.
    
    Handles normalization, weighting, and score calculation with
    detailed per-place telemetry for A/B testing measurement.
    """
    
    def __init__(
        self,
        weights: Optional[WeightConfig] = None,
        enable_telemetry: bool = True,
    ):
        """
        Initialize the scorer.
        
        Args:
            weights: Weight configuration to use. If None, uses defaults
            enable_telemetry: Whether to log detailed telemetry
        """
        self.weights = weights or DEFAULT_WEIGHTS
        self.enable_telemetry = enable_telemetry
        self.telemetry_log: List[ScoringTelemetry] = []
    
    def score_places(
        self,
        places_df: pd.DataFrame,
        etas_sec: Dict[str, int],
        hex_df: pd.DataFrame,
        user_preferences: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Score places with comprehensive telemetry.
        
        Args:
            places_df: DataFrame with columns: id, name, lat, lng, rating, nratings, 
                       open_now, primary, types
            etas_sec: Dict mapping place_id to travel time in seconds
            hex_df: DataFrame with hex-level aggregated features (localness, cluster)
            user_preferences: Optional dict mapping type tokens to affinity multipliers
        
        Returns:
            DataFrame with added columns: score, score breakdown components, telemetry
        """
        df = places_df.copy()
        
        # Add ETAs
        df["eta"] = df["id"].map(etas_sec).fillna(np.nan)
        
        # Calculate raw crowd proxy from ratings count
        df["crowd_proxy"] = df["nratings"].fillna(0).astype(float)
        
        # Normalize core metrics
        df["rating_norm"] = normalize_rating(df["rating"].fillna(0).to_numpy())
        df["eta_norm"] = normalize_eta(
            df["eta"].fillna(df["eta"].max() or 600).to_numpy(),
            min_eta_sec=180
        )
        df["crowd_norm"] = normalize_crowd_proxy(df["crowd_proxy"].to_numpy())
        df["open_now"] = df["open_now"].fillna(0).astype(float)
        
        # Join hex-level features
        h3_res = int(hex_df.attrs.get("h3_res", 9))
        df["hex"] = df.apply(
            lambda r: h3.geo_to_h3(r["lat"], r["lng"], h3_res),
            axis=1
        )
        
        hex_map = hex_df.set_index("hex")
        df["localness"] = df["hex"].map(hex_map["localness"]).fillna(0)
        df["localness_norm"] = normalize_localness(df["localness"].to_numpy())
        df["cluster_id"] = df["hex"].map(hex_map.get("cluster", pd.Series())).fillna(-1).astype(int)
        
        # Calculate diversity gain
        df["diversity_gain"] = self._calculate_diversity_gain(df)
        
        # Apply user preferences
        df["pref_mult"] = self._apply_user_preferences(df, user_preferences)
        
        # Calculate weighted score
        w = self.weights.to_dict()
        
        # Component scores (already weighted)
        df["rating_score"] = w["w_rating"] * df["rating_norm"]
        df["diversity_score"] = w["w_diversity"] * df["diversity_gain"]
        df["eta_score"] = w["w_eta"] * df["eta_norm"]  # Already inverted in normalize_eta
        df["open_score"] = w["w_open"] * df["open_now"]
        df["crowd_score"] = w["w_crowd"] * df["crowd_norm"]  # Penalty (subtract below)
        
        # Final score with preference multiplier
        df["score"] = (
            df["rating_score"] +
            df["diversity_score"] +
            df["eta_score"] +
            df["open_score"] -
            df["crowd_score"]  # Crowd is a penalty
        ) * df["pref_mult"]
        
        # Log telemetry
        if self.enable_telemetry:
            self._log_telemetry(df)
        
        return df
    
    def _calculate_diversity_gain(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate diversity gain for each place.
        
        Diversity represents how unique/rare a place type is.
        Uses inverse popularity as a proxy.
        """
        # Count type occurrences
        all_types = "|".join(df["types"].fillna(""))
        type_counts = pd.Series(all_types.split("|")).value_counts()
        
        # Calculate rarity (inverse popularity)
        rarity = type_counts.max() - type_counts
        rarity_norm = (rarity - rarity.min()) / (rarity.max() - rarity.min() + 1e-9)
        
        # Average rarity for each place's types
        def avg_rarity(types_str: str) -> float:
            if not types_str:
                return 0.0
            tokens = [t for t in types_str.split("|") if t]
            if not tokens:
                return 0.0
            return float(np.mean([rarity_norm.get(t, 0.0) for t in tokens]))
        
        return df["types"].fillna("").apply(avg_rarity)
    
    def _apply_user_preferences(
        self,
        df: pd.DataFrame,
        preferences: Optional[Dict[str, float]]
    ) -> pd.Series:
        """
        Apply user preference multipliers based on place types.
        
        Args:
            df: DataFrame with 'types' column
            preferences: Dict mapping type tokens to affinity values
                        (e.g., {"restaurant": 0.2, "museum": -0.1})
        
        Returns:
            Series of preference multipliers (typically 0.5 to 1.5)
        """
        if not preferences:
            return pd.Series(1.0, index=df.index)
        
        def calc_multiplier(types_str: str) -> float:
            if not types_str:
                return 1.0
            
            for token in types_str.split("|"):
                if token in preferences:
                    # Convert affinity to multiplier: affinity ∈ [-1, 1] → mult ∈ [0.5, 1.5]
                    affinity = preferences[token]
                    return max(0.5, min(1.5, 1.0 + affinity))
            
            return 1.0
        
        return df["types"].fillna("").apply(calc_multiplier)
    
    def _log_telemetry(self, df: pd.DataFrame):
        """
        Log detailed telemetry for each place.
        
        Stores telemetry in self.telemetry_log for later analysis.
        """
        for row in df.itertuples():
            breakdown = ScoreBreakdown(
                rating_score=float(row.rating_score),
                diversity_score=float(row.diversity_score),
                eta_score=float(row.eta_score),
                open_score=float(row.open_score),
                crowd_score=float(row.crowd_score),
                preference_multiplier=float(row.pref_mult),
                final_score=float(row.score),
            )
            
            telemetry = ScoringTelemetry(
                place_id=row.id,
                place_name=row.name,
                variant_name=self.weights.variant_name,
                weights=self.weights.to_dict(),
                breakdown=breakdown,
                eta_sec=0 if pd.isna(row.eta) else int(row.eta),
                is_open_now=bool(row.open_now),
                cluster_id=None if row.cluster_id == -1 else int(row.cluster_id),
                cluster_label=None,  # Will be filled in by caller
                raw_values={
                    "rating": None if pd.isna(row.rating) else float(row.rating),
                    "nratings": int(row.nratings) if hasattr(row, "nratings") else 0,
                    "localness": float(row.localness) if hasattr(row, "localness") else 0.0,
                    "diversity_gain": float(row.diversity_gain),
                    "crowd_proxy": float(row.crowd_proxy) if hasattr(row, "crowd_proxy") else 0.0,
                }
            )
            
            self.telemetry_log.append(telemetry)
            
            # Log to logger for real-time monitoring
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Scoring telemetry: {telemetry.to_json()}")
    
    def get_telemetry(self) -> List[ScoringTelemetry]:
        """Get all collected telemetry."""
        return self.telemetry_log
    
    def clear_telemetry(self):
        """Clear telemetry log."""
        self.telemetry_log.clear()
    
    def export_telemetry_json(self, filepath: str):
        """
        Export telemetry to JSON file for analysis.
        
        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(
                [t.to_dict() for t in self.telemetry_log],
                f,
                ensure_ascii=False,
                indent=2
            )


# Convenience function for backward compatibility
def score_places(
    places_df: pd.DataFrame,
    etas_sec: Dict[str, int],
    hex_df: pd.DataFrame,
    weights: Optional[WeightConfig] = None,
    user_preferences: Optional[Dict[str, float]] = None,
    enable_telemetry: bool = True,
) -> pd.DataFrame:
    """
    Score places using the default scorer.
    
    Convenience function that creates a PlaceScorer and runs scoring.
    
    Args:
        places_df: DataFrame with place data
        etas_sec: Travel times mapping
        hex_df: Hex-level features
        weights: Optional weight configuration
        user_preferences: Optional user preference multipliers
        enable_telemetry: Whether to log telemetry
    
    Returns:
        DataFrame with scores and breakdown
    """
    scorer = PlaceScorer(weights=weights, enable_telemetry=enable_telemetry)
    return scorer.score_places(places_df, etas_sec, hex_df, user_preferences)
