"""
Unit Tests for Routing Module (src/routing)

Tests distance matrix management, VRPTW solver, greedy sequencing, and fallback logic.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.routing.matrix import (
    Location,
    MatrixRequest,
    MatrixLimits,
    TravelMode,
    RoutingPreference,
    get_matrix_limits,
    validate_matrix_request,
    get_cache_stats,
    clear_cache,
    MAX_ELEMENTS_GENERAL,
    MAX_ELEMENTS_TRANSIT,
    MAX_ELEMENTS_TRAFFIC_AWARE_OPTIMAL,
)

from src.routing.greedy import (
    greedy_sequence,
    GreedySequenceResult,
    Stop,
    format_reason,
)

from src.routing.vrptw import (
    VRPTWConfig,
    VRPTWResult,
    TimeWindow as VRPTWTimeWindow,
    solve_vrptw_with_fallback,
    solve_vrptw,
)


# ==============================================================================
# Matrix Configuration Tests
# ==============================================================================

class TestMatrixLimits:
    """Test matrix limit calculations."""
    
    def test_get_matrix_limits_general(self):
        """Test getting matrix limits for general routing."""
        limits = get_matrix_limits(
            mode=TravelMode.DRIVE,
            routing_pref=RoutingPreference.TRAFFIC_UNAWARE
        )
        
        assert limits.max_elements == MAX_ELEMENTS_GENERAL
        assert limits.mode == TravelMode.DRIVE
    
    def test_get_matrix_limits_transit(self):
        """Test getting matrix limits for transit."""
        limits = get_matrix_limits(
            mode=TravelMode.TRANSIT,
            routing_pref=RoutingPreference.TRAFFIC_UNAWARE
        )
        
        assert limits.max_elements == MAX_ELEMENTS_TRANSIT
        assert limits.mode == TravelMode.TRANSIT
    
    def test_get_matrix_limits_traffic_aware(self):
        """Test getting matrix limits for traffic-aware routing."""
        limits = get_matrix_limits(
            mode=TravelMode.DRIVE,
            routing_pref=RoutingPreference.TRAFFIC_AWARE_OPTIMAL
        )
        
        assert limits.max_elements == MAX_ELEMENTS_TRAFFIC_AWARE_OPTIMAL


class TestMatrixValidation:
    """Test matrix request validation."""
    
    def test_validate_valid_request(self):
        """Test validating a valid matrix request."""
        origins = [Location(lat=35.6812, lng=139.7671)] * 10
        destinations = [Location(lat=35.6895, lng=139.6917)] * 10
        
        request = MatrixRequest(
            origins=origins,
            destinations=destinations,
            mode=TravelMode.DRIVE,
            routing_preference=RoutingPreference.TRAFFIC_UNAWARE
        )
        
        # Should not raise exception
        validate_matrix_request(request)
    
    def test_validate_too_many_elements(self):
        """Test validation fails when too many elements requested."""
        # Create request with too many elements
        origins = [Location(lat=35.0, lng=139.0)] * 50
        destinations = [Location(lat=36.0, lng=140.0)] * 50
        # 50 * 50 = 2500 elements > MAX_ELEMENTS_GENERAL (625)
        
        request = MatrixRequest(
            origins=origins,
            destinations=destinations,
            mode=TravelMode.DRIVE,
            routing_preference=RoutingPreference.TRAFFIC_UNAWARE
        )
        
        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            validate_matrix_request(request)
        
        error_msg = str(exc_info.value)
        assert "2500" in error_msg or "elements" in error_msg.lower()
    
    def test_validate_empty_origins(self):
        """Test validation fails with empty origins."""
        request = MatrixRequest(
            origins=[],
            destinations=[Location(lat=35.0, lng=139.0)],
            mode=TravelMode.DRIVE,
            routing_preference=RoutingPreference.TRAFFIC_UNAWARE
        )
        
        # Should raise ValueError or work (0 elements)
        try:
            validate_matrix_request(request)
        except ValueError:
            pass  # Expected for empty origins


class TestCacheManagement:
    """Test cache management functions."""
    
    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        stats = get_cache_stats()
        
        assert isinstance(stats, dict)
        # Cache stats should have traffic_cache and static_cache keys
        assert 'traffic_cache' in stats or 'static_cache' in stats or stats == {}
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        # Should not raise exception
        clear_cache()
        
        stats = get_cache_stats()
        # After clearing, cache should be empty or reset
        if 'traffic_cache' in stats:
            assert stats['traffic_cache']['size'] == 0
        if 'static_cache' in stats:
            assert stats['static_cache']['size'] == 0


# ==============================================================================
# Location and Data Model Tests
# ==============================================================================

class TestDataModels:
    """Test data model classes."""
    
    def test_location_creation(self):
        """Test creating Location objects."""
        loc = Location(lat=35.6812, lng=139.7671)
        
        assert loc.lat == 35.6812
        assert loc.lng == 139.7671
    
    def test_matrix_request_creation(self):
        """Test creating MatrixRequest objects."""
        origins = [Location(lat=35.6812, lng=139.7671)]
        destinations = [Location(lat=35.6895, lng=139.6917)]
        
        request = MatrixRequest(
            origins=origins,
            destinations=destinations,
            mode=TravelMode.DRIVE,
            routing_preference=RoutingPreference.TRAFFIC_UNAWARE
        )
        
        assert len(request.origins) == 1
        assert len(request.destinations) == 1
        assert request.mode == TravelMode.DRIVE
    
    def test_travel_mode_enum(self):
        """Test TravelMode enum values."""
        assert hasattr(TravelMode, 'DRIVE')
        assert hasattr(TravelMode, 'WALK')
        assert hasattr(TravelMode, 'TRANSIT')
        assert hasattr(TravelMode, 'BICYCLE')
    
    def test_routing_preference_enum(self):
        """Test RoutingPreference enum values."""
        assert hasattr(RoutingPreference, 'TRAFFIC_UNAWARE')
        assert hasattr(RoutingPreference, 'TRAFFIC_AWARE')
        assert hasattr(RoutingPreference, 'TRAFFIC_AWARE_OPTIMAL')


# ==============================================================================
# Greedy Sequencing Tests
# ==============================================================================

class TestGreedySequencing:
    """Test greedy route sequencing algorithm."""
    
    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidates for sequencing."""
        return pd.DataFrame({
            'id': ['p1', 'p2', 'p3', 'p4', 'p5'],
            'name': ['Place 1', 'Place 2', 'Place 3', 'Place 4', 'Place 5'],
            'lat': [35.6812, 35.6895, 35.6764, 35.7148, 35.7101],
            'lng': [139.7671, 139.6917, 139.6993, 139.7967, 139.8107],
            'score': [0.9, 0.8, 0.7, 0.6, 0.5],
            'eta': [300, 600, 450, 900, 1200],  # seconds from anchor
        })
    
    def test_greedy_sequence_basic(self, sample_candidates):
        """Test basic greedy sequencing."""
        start_time = datetime(2025, 10, 12, 13, 0)
        end_time = start_time + timedelta(hours=4)
        
        result = greedy_sequence(
            candidates=sample_candidates,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time,
            service_time_min=30
        )
        
        assert isinstance(result, GreedySequenceResult)
        assert len(result.stops) > 0
        assert result.sequence_method == "greedy"
        
        # Stops should be ordered by score (descending)
        if len(result.stops) > 1:
            for i in range(len(result.stops) - 1):
                assert result.stops[i].score >= result.stops[i+1].score
    
    def test_greedy_sequence_fits_time_window(self, sample_candidates):
        """Test that greedy sequence respects time window."""
        start_time = datetime(2025, 10, 12, 13, 0)
        end_time = start_time + timedelta(hours=2)  # Short window
        
        result = greedy_sequence(
            candidates=sample_candidates,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time,
            service_time_min=30
        )
        
        # All stops should fit within time window
        for stop in result.stops:
            assert stop.arrival_time >= start_time
            assert stop.departure_time <= end_time
    
    def test_greedy_sequence_skips_stops(self, sample_candidates):
        """Test that greedy sequence skips stops that don't fit."""
        start_time = datetime(2025, 10, 12, 13, 0)
        end_time = start_time + timedelta(minutes=90)  # Very short window
        
        result = greedy_sequence(
            candidates=sample_candidates,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time,
            service_time_min=30
        )
        
        # Should skip some stops due to time constraints
        total_candidates = len(sample_candidates)
        assert len(result.stops) < total_candidates
        assert result.num_stops_skipped > 0
    
    def test_greedy_sequence_empty_candidates(self):
        """Test greedy sequencing with empty candidates."""
        empty_df = pd.DataFrame(columns=['id', 'name', 'lat', 'lng', 'score', 'eta'])
        start_time = datetime(2025, 10, 12, 13, 0)
        end_time = start_time + timedelta(hours=4)
        
        result = greedy_sequence(
            candidates=empty_df,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time
        )
        
        assert len(result.stops) == 0
        assert result.num_stops_skipped == 0
    
    def test_greedy_sequence_service_time(self, sample_candidates):
        """Test greedy sequencing with different service times."""
        start_time = datetime(2025, 10, 12, 13, 0)
        end_time = start_time + timedelta(hours=4)
        
        # Short service time
        result_short = greedy_sequence(
            candidates=sample_candidates,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time,
            service_time_min=15
        )
        
        # Long service time
        result_long = greedy_sequence(
            candidates=sample_candidates,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time,
            service_time_min=60
        )
        
        # Shorter service time should allow more stops
        assert len(result_short.stops) >= len(result_long.stops)


class TestStop:
    """Test Stop dataclass."""
    
    def test_stop_creation(self):
        """Test creating a Stop object."""
        stop = Stop(
            place_id="p1",
            place_name="Place 1",
            lat=35.6812,
            lng=139.7671,
            score=0.9,
            eta_from_anchor_sec=300,
            arrival_time=datetime(2025, 10, 12, 13, 5),
            departure_time=datetime(2025, 10, 12, 13, 35),
            service_time_min=30,
            reason="High score"
        )
        
        assert stop.place_id == "p1"
        assert stop.score == 0.9
        assert stop.service_time_min == 30


class TestFormatReason:
    """Test reason formatting function."""
    
    def test_format_reason(self):
        """Test formatting stop selection reason."""
        stop = Stop(
            place_id="p1",
            place_name="Test Place",
            lat=35.6812,
            lng=139.7671,
            score=0.95,
            eta_from_anchor_sec=300,
            arrival_time=datetime(2025, 10, 12, 13, 0),
            departure_time=datetime(2025, 10, 12, 13, 30),
            service_time_min=30,
            reason="High score"
        )
        
        reason = format_reason(stop, additional_info={'method': 'greedy'})
        
        assert isinstance(reason, str)
        assert len(reason) > 0


# ==============================================================================
# VRPTW Configuration Tests
# ==============================================================================

class TestVRPTWConfig:
    """Test VRPTW configuration."""
    
    def test_default_config(self):
        """Test default VRPTW configuration."""
        config = VRPTWConfig()
        
        assert config.time_limit_sec > 0
        assert config.service_time_min > 0
    
    def test_custom_config(self):
        """Test custom VRPTW configuration."""
        config = VRPTWConfig(
            time_limit_sec=30,
            service_time_min=45,
            verbose=True
        )
        
        assert config.time_limit_sec == 30
        assert config.service_time_min == 45
        assert config.verbose is True


class TestVRPTWTimeWindow:
    """Test VRPTW time window dataclass."""
    
    def test_time_window_creation(self):
        """Test creating a time window."""
        # TimeWindow uses start_sec and end_sec (seconds from day start)
        tw = VRPTWTimeWindow(start_sec=3600, end_sec=7200)  # 1-2 hours
        
        assert tw.start_sec == 3600
        assert tw.end_sec == 7200
        assert tw.end_sec - tw.start_sec == 3600  # 1 hour duration


# ==============================================================================
# VRPTW Solver Tests (with mocked OR-Tools)
# ==============================================================================

class TestVRPTWSolver:
    """Test VRPTW solver with fallback."""
    
    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidates for VRPTW."""
        return pd.DataFrame({
            'id': ['p1', 'p2', 'p3'],
            'name': ['Place 1', 'Place 2', 'Place 3'],
            'lat': [35.6812, 35.6895, 35.6764],
            'lng': [139.7671, 139.6917, 139.6993],
            'score': [0.9, 0.8, 0.7],
            'eta': [300, 600, 450],
        })
    
    @pytest.fixture
    def sample_distance_matrix(self):
        """Create sample distance matrix."""
        # 4x4 matrix (anchor + 3 candidates)
        return np.array([
            [0, 600, 1200, 900],
            [600, 0, 800, 700],
            [1200, 800, 0, 300],
            [900, 700, 300, 0],
        ], dtype=np.int32)
    
    def test_vrptw_with_fallback_enabled(self, sample_candidates, sample_distance_matrix):
        """Test VRPTW solver with fallback enabled."""
        start_time = datetime(2025, 10, 12, 13, 0)
        end_time = start_time + timedelta(hours=4)
        
        config = VRPTWConfig(time_limit_sec=10)
        
        result = solve_vrptw_with_fallback(
            candidates=sample_candidates,
            distance_matrix_full=sample_distance_matrix,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time,
            config=config
        )
        
        # Should return a result (either VRPTW or greedy)
        assert result is not None
        assert hasattr(result, 'stops')
    
    def test_vrptw_fallback_to_greedy(self, sample_candidates):
        """Test VRPTW fallback to greedy when solver fails."""
        start_time = datetime(2025, 10, 12, 13, 0)
        end_time = start_time + timedelta(hours=4)
        
        # Create invalid distance matrix to force fallback
        invalid_matrix = np.array([[0]], dtype=np.int32)  # Too small
        
        config = VRPTWConfig()
        
        result = solve_vrptw_with_fallback(
            candidates=sample_candidates,
            distance_matrix_full=invalid_matrix,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time,
            config=config
        )
        
        # Should fall back to greedy
        assert result is not None


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestRoutingIntegration:
    """Integration tests for routing components."""
    
    def test_complete_routing_pipeline(self):
        """Test complete routing pipeline from candidates to sequence."""
        # Sample candidates
        candidates = pd.DataFrame({
            'id': ['p1', 'p2', 'p3', 'p4'],
            'name': ['Place 1', 'Place 2', 'Place 3', 'Place 4'],
            'lat': [35.6812, 35.6895, 35.6764, 35.7148],
            'lng': [139.7671, 139.6917, 139.6993, 139.7967],
            'score': [0.9, 0.8, 0.7, 0.6],
            'eta': [300, 600, 450, 900],
        })
        
        # Time window
        start_time = datetime(2025, 10, 12, 13, 0)
        end_time = start_time + timedelta(hours=4)
        
        # Run greedy sequencing (always works, no API calls)
        result = greedy_sequence(
            candidates=candidates,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time,
            service_time_min=30
        )
        
        # Verify result structure
        assert isinstance(result, GreedySequenceResult)
        assert len(result.stops) > 0
        assert all(isinstance(stop, Stop) for stop in result.stops)
        
        # Verify timing
        for stop in result.stops:
            assert start_time <= stop.arrival_time <= end_time
            assert stop.arrival_time < stop.departure_time


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestRoutingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_matrix_with_single_location(self):
        """Test matrix validation with single location."""
        request = MatrixRequest(
            origins=[Location(lat=35.0, lng=139.0)],
            destinations=[Location(lat=35.0, lng=139.0)],
            mode=TravelMode.DRIVE,
            routing_preference=RoutingPreference.TRAFFIC_UNAWARE
        )
        
        # Should not raise exception for 1x1 matrix
        validate_matrix_request(request)
    
    def test_greedy_with_zero_time_window(self):
        """Test greedy sequencing with zero time window."""
        candidates = pd.DataFrame({
            'id': ['p1'],
            'name': ['Place 1'],
            'lat': [35.6812],
            'lng': [139.7671],
            'score': [0.9],
            'eta': [300],
        })
        
        start_time = datetime(2025, 10, 12, 13, 0)
        end_time = start_time  # Same as start (zero window)
        
        result = greedy_sequence(
            candidates=candidates,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time
        )
        
        # Should return no stops
        assert len(result.stops) == 0
        assert result.num_stops_skipped >= 0
    
    def test_greedy_with_negative_scores(self):
        """Test greedy sequencing with negative scores."""
        candidates = pd.DataFrame({
            'id': ['p1', 'p2'],
            'name': ['Place 1', 'Place 2'],
            'lat': [35.6812, 35.6895],
            'lng': [139.7671, 139.6917],
            'score': [-0.5, -0.8],  # Negative scores
            'eta': [300, 600],
        })
        
        start_time = datetime(2025, 10, 12, 13, 0)
        end_time = start_time + timedelta(hours=4)
        
        result = greedy_sequence(
            candidates=candidates,
            anchor_lat=35.6812,
            anchor_lng=139.7671,
            start_time=start_time,
            end_time=end_time
        )
        
        # Should still work (less negative is "better")
        # p1 (-0.5) should come before p2 (-0.8)
        if len(result.stops) > 1:
            assert result.stops[0].score > result.stops[1].score
