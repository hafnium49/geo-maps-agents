"""
Unit Tests for Scoring Module (src/scoring)

Tests normalization, diversity scoring, preference multipliers, and the
complete PlaceScorer pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import h3

from src.scoring.normalization import (
    percentile_norm,
    normalize_rating,
    normalize_eta,
    normalize_crowd_proxy,
    normalize_diversity,
    normalize_localness,
    robust_norm,
)

from src.scoring.weights import WeightConfig, DEFAULT_WEIGHTS, select_ab_variant

from src.scoring.scorer import PlaceScorer, ScoreBreakdown, ScoringTelemetry, score_places


# ==============================================================================
# Normalization Tests
# ==============================================================================

class TestPercentileNorm:
    """Test percentile-based normalization."""
    
    def test_basic_normalization(self):
        """Test basic percentile normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        normalized = percentile_norm(data, low_percentile=10, high_percentile=90)
        
        # Check range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        
        # Middle values should be between 0 and 1
        assert 0.0 < normalized[4] < 1.0
    
    def test_inverted_normalization(self):
        """Test inverted normalization (lower is better)."""
        data = np.array([100, 200, 300, 400, 500])
        normalized = percentile_norm(data, invert=True)
        
        # Lower values should get higher scores
        assert normalized[0] > normalized[-1]
        assert normalized[1] > normalized[3]
    
    def test_degenerate_case_all_same(self):
        """Test normalization when all values are identical."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        normalized = percentile_norm(data)
        
        # Should return all zeros for degenerate case
        assert np.all(normalized == 0.0)
    
    def test_empty_array(self):
        """Test normalization with empty array."""
        data = np.array([])
        normalized = percentile_norm(data)
        
        assert len(normalized) == 0
    
    def test_nan_handling(self):
        """Test normalization with NaN values."""
        data = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        normalized = percentile_norm(data)
        
        # Should handle NaN values gracefully
        assert len(normalized) == len(data)
        assert not np.all(np.isnan(normalized))


class TestNormalizeRating:
    """Test rating normalization."""
    
    def test_typical_ratings(self):
        """Test normalization with typical rating values."""
        ratings = np.array([3.5, 4.0, 4.2, 4.5, 4.7, 5.0, 2.8, 3.9])
        normalized = normalize_rating(ratings)
        
        # Check range
        assert np.all((normalized >= 0.0) & (normalized <= 1.0))
        
        # Higher ratings should have higher scores
        assert normalized[5] >= normalized[0]  # 5.0 >= 3.5
    
    def test_edge_case_low_ratings(self):
        """Test with very low ratings."""
        ratings = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        normalized = normalize_rating(ratings)
        
        # Should still normalize correctly
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_missing_ratings(self):
        """Test with NaN ratings."""
        ratings = np.array([4.5, np.nan, 4.0, np.nan, 5.0])
        normalized = normalize_rating(ratings)
        
        assert len(normalized) == len(ratings)


class TestNormalizeEta:
    """Test ETA (travel time) normalization."""
    
    def test_typical_etas(self):
        """Test with typical ETA values."""
        etas = np.array([300, 600, 900, 1200, 1500])  # 5, 10, 15, 20, 25 min
        normalized = normalize_eta(etas, min_eta_sec=180)
        
        # Check range
        assert np.all((normalized >= 0.0) & (normalized <= 1.0))
        
        # Lower ETAs should have higher scores (inverted)
        assert normalized[0] > normalized[-1]
    
    def test_minimum_eta_clamping(self):
        """Test minimum ETA clamping."""
        etas = np.array([60, 120, 180, 240, 300])  # Some below min
        normalized = normalize_eta(etas, min_eta_sec=180)
        
        # Values below min should be clamped
        # After clamping: [180, 180, 180, 240, 300]
        # After inversion, first three should have same score
        assert normalized[0] == normalized[2]
    
    def test_very_short_etas(self):
        """Test with very short ETAs (walking distance)."""
        etas = np.array([30, 60, 90, 120])  # Very short times
        normalized = normalize_eta(etas, min_eta_sec=60)
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


# ==============================================================================
# Weight Configuration Tests
# ==============================================================================

class TestWeightConfig:
    """Test weight configuration management."""
    
    def test_default_weights(self):
        """Test default weight configuration."""
        assert DEFAULT_WEIGHTS.w_rating == 0.30
        assert DEFAULT_WEIGHTS.w_diversity == 0.25
        assert DEFAULT_WEIGHTS.w_eta == 0.20
        assert DEFAULT_WEIGHTS.w_open == 0.15
        assert DEFAULT_WEIGHTS.w_crowd == 0.10
    
    def test_custom_weights(self):
        """Test custom weight configuration."""
        custom = WeightConfig(
            w_rating=0.40,
            w_diversity=0.20,
            w_eta=0.15,
            w_open=0.15,
            w_crowd=0.10,
            variant_name="custom_test"
        )
        
        assert custom.w_rating == 0.40
        assert custom.variant_name == "custom_test"
    
    def test_weights_to_dict(self):
        """Test weight config serialization to dict."""
        weights = WeightConfig(w_rating=0.35, w_diversity=0.25)
        weights_dict = weights.to_dict()
        
        assert "w_rating" in weights_dict
        assert weights_dict["w_rating"] == 0.35
        # to_dict() only returns weight values, not variant_name
        assert len(weights_dict) == 5  # 5 weight parameters


class TestABVariantSelection:
    """Test A/B variant selection."""
    
    def test_variant_selection_deterministic(self):
        """Test that variant selection is deterministic for same user."""
        user_id = "test_user_123"
        
        # Same user should get same variant
        variant1 = select_ab_variant(user_id)
        variant2 = select_ab_variant(user_id)
        
        assert variant1.variant_name == variant2.variant_name
    
    def test_variant_selection_different_users(self):
        """Test that different users can get different variants."""
        users = [f"user_{i}" for i in range(100)]
        variants = [select_ab_variant(u).variant_name for u in users]
        
        # Should have some distribution (not all same)
        unique_variants = set(variants)
        assert len(unique_variants) > 1  # At least 2 different variants


# ==============================================================================
# PlaceScorer Tests
# ==============================================================================

class TestPlaceScorer:
    """Test the main PlaceScorer class."""
    
    def test_scorer_initialization(self):
        """Test scorer initialization."""
        scorer = PlaceScorer()
        
        assert scorer.weights == DEFAULT_WEIGHTS
        assert scorer.enable_telemetry is True
        assert len(scorer.telemetry_log) == 0
    
    def test_score_places_basic(self, sample_places_for_scoring, sample_etas, sample_hex_df):
        """Test basic place scoring."""
        scorer = PlaceScorer()
        scored_df = scorer.score_places(
            sample_places_for_scoring,
            sample_etas,
            sample_hex_df
        )
        
        # Check that score column exists
        assert 'score' in scored_df.columns
        
        # Check that all scores are computed
        assert not scored_df['score'].isna().any()
        
        # Check score components exist
        assert 'rating_score' in scored_df.columns
        assert 'diversity_score' in scored_df.columns
        assert 'eta_score' in scored_df.columns
        assert 'open_score' in scored_df.columns
        assert 'crowd_score' in scored_df.columns
    
    def test_score_places_telemetry(self, sample_places_for_scoring, sample_etas, sample_hex_df):
        """Test telemetry logging."""
        scorer = PlaceScorer(enable_telemetry=True)
        scored_df = scorer.score_places(
            sample_places_for_scoring,
            sample_etas,
            sample_hex_df
        )
        
        # Check telemetry was logged
        telemetry = scorer.get_telemetry()
        assert len(telemetry) == len(sample_places_for_scoring)
        
        # Check telemetry structure
        first_entry = telemetry[0]
        assert isinstance(first_entry, ScoringTelemetry)
        assert first_entry.place_id == 'p1'
        assert first_entry.place_name == 'Place 1'
        assert isinstance(first_entry.breakdown, ScoreBreakdown)
    
    def test_score_places_no_telemetry(self, sample_places_for_scoring, sample_etas, sample_hex_df):
        """Test scoring without telemetry."""
        scorer = PlaceScorer(enable_telemetry=False)
        scored_df = scorer.score_places(
            sample_places_for_scoring,
            sample_etas,
            sample_hex_df
        )
        
        # Telemetry should be empty
        assert len(scorer.get_telemetry()) == 0
    
    def test_user_preferences(self, sample_places_for_scoring, sample_etas, sample_hex_df):
        """Test user preference multipliers."""
        scorer = PlaceScorer()
        
        # Boost restaurants, penalize museums
        preferences = {
            'restaurant': 0.3,  # +30% boost
            'museum': -0.2,     # -20% penalty
        }
        
        scored_df = scorer.score_places(
            sample_places_for_scoring,
            sample_etas,
            sample_hex_df,
            user_preferences=preferences
        )
        
        # Check preference multipliers were applied
        assert 'pref_mult' in scored_df.columns
        
        # Restaurant should have multiplier > 1.0
        p1_mult = scored_df[scored_df['id'] == 'p1']['pref_mult'].iloc[0]
        assert p1_mult > 1.0
        
        # Museum should have multiplier < 1.0
        p4_mult = scored_df[scored_df['id'] == 'p4']['pref_mult'].iloc[0]
        assert p4_mult < 1.0
    
    def test_diversity_gain_calculation(self, sample_places_for_scoring, sample_etas, sample_hex_df):
        """Test diversity gain calculation."""
        scorer = PlaceScorer()
        scored_df = scorer.score_places(
            sample_places_for_scoring,
            sample_etas,
            sample_hex_df
        )
        
        # Check diversity gain was calculated
        assert 'diversity_gain' in scored_df.columns
        
        # Rare types should have higher diversity gain
        # (sushi is rarer than generic restaurant)
        p3_diversity = scored_df[scored_df['id'] == 'p3']['diversity_gain'].iloc[0]
        assert p3_diversity >= 0.0
    
    def test_clear_telemetry(self, sample_places_for_scoring, sample_etas, sample_hex_df):
        """Test telemetry clearing."""
        scorer = PlaceScorer()
        scorer.score_places(sample_places_for_scoring, sample_etas, sample_hex_df)
        
        assert len(scorer.telemetry_log) > 0
        
        scorer.clear_telemetry()
        assert len(scorer.telemetry_log) == 0
    
    def test_custom_weights(self, sample_places_for_scoring, sample_etas, sample_hex_df):
        """Test scoring with custom weights."""
        custom_weights = WeightConfig(
            w_rating=0.50,  # Heavily weight rating
            w_diversity=0.10,
            w_eta=0.10,
            w_open=0.20,
            w_crowd=0.10,
            variant_name="rating_focused"
        )
        
        scorer = PlaceScorer(weights=custom_weights)
        scored_df = scorer.score_places(
            sample_places_for_scoring,
            sample_etas,
            sample_hex_df
        )
        
        # Check that rating component is larger
        assert scored_df['rating_score'].mean() > scored_df['diversity_score'].mean()


# ==============================================================================
# Score Breakdown Tests
# ==============================================================================

class TestScoreBreakdown:
    """Test ScoreBreakdown dataclass."""
    
    def test_score_breakdown_creation(self):
        """Test creating a score breakdown."""
        breakdown = ScoreBreakdown(
            rating_score=0.15,
            diversity_score=0.10,
            eta_score=0.08,
            open_score=0.15,
            crowd_score=0.05,
            preference_multiplier=1.0,
            final_score=0.43
        )
        
        assert breakdown.rating_score == 0.15
        assert breakdown.final_score == 0.43
    
    def test_score_breakdown_to_dict(self):
        """Test converting breakdown to dict."""
        breakdown = ScoreBreakdown(
            rating_score=0.15,
            diversity_score=0.10,
            eta_score=0.08,
            open_score=0.15,
            crowd_score=0.05,
            preference_multiplier=1.0,
            final_score=0.43
        )
        
        breakdown_dict = breakdown.to_dict()
        assert isinstance(breakdown_dict, dict)
        assert breakdown_dict['rating_score'] == 0.15


# ==============================================================================
# Convenience Function Tests
# ==============================================================================

class TestScorePlacesFunction:
    """Test the convenience score_places() function."""
    
    def test_score_places_function(self, sample_places_for_scoring, sample_etas, sample_hex_df):
        """Test convenience function."""
        scored_df = score_places(
            sample_places_for_scoring,
            sample_etas,
            sample_hex_df
        )
        
        assert 'score' in scored_df.columns
        assert not scored_df['score'].isna().any()


# ==============================================================================
# Edge Case Tests
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_places_df(self):
        """Test scoring with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['id', 'name', 'lat', 'lng', 'rating', 'nratings', 'open_now', 'primary', 'types'])
        etas = {}
        hex_df = pd.DataFrame(columns=['hex', 'localness', 'cluster'])
        hex_df.attrs['h3_res'] = 9
        
        scorer = PlaceScorer()
        scored_df = scorer.score_places(empty_df, etas, hex_df)
        
        # Should handle gracefully
        assert len(scored_df) == 0
    
    def test_missing_etas(self, sample_places_for_scoring, sample_hex_df):
        """Test scoring with missing ETAs."""
        etas = {'p1': 300}  # Only one ETA
        
        scorer = PlaceScorer()
        scored_df = scorer.score_places(sample_places_for_scoring, etas, sample_hex_df)
        
        # Should handle missing ETAs (fill with NaN)
        assert 'score' in scored_df.columns
    
    def test_all_closed_places(self, sample_places_for_scoring, sample_etas, sample_hex_df):
        """Test scoring when all places are closed."""
        sample_places_for_scoring['open_now'] = 0  # All closed
        
        scorer = PlaceScorer()
        scored_df = scorer.score_places(sample_places_for_scoring, sample_etas, sample_hex_df)
        
        # Should still score, but open_score should be 0
        assert scored_df['open_score'].sum() == 0.0
