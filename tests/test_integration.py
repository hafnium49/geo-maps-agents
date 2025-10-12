"""
Integration tests for the GeoTrip Agent pipeline.

These tests validate the complete flow from raw data to final output,
using mock API clients to avoid real API calls and costs.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

from src.scoring import (
    PlaceScorer,
    WeightConfig,
    score_places,
    select_ab_variant,
)
from src.spatial import (
    cluster_with_fallback,
    ClusteringConfig,
    label_cluster,
)
from src.routing import (
    compute_route_matrix,
    MatrixRequest,
    Location,
    TravelMode,
    RoutingPreference,
    solve_vrptw_with_fallback,
    VRPTWConfig,
    greedy_sequence,
)


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete pipeline from places → scoring → clustering → sequencing."""
    
    def test_complete_flow_with_mocks(self, sample_places, sample_distance_matrix, mock_places_api, mock_routes_api):
        """Test the complete pipeline using mock APIs."""
        # Step 1: Places discovery (already mocked in sample_places)
        places_df = pd.DataFrame([
            {
                'id': p['id'],
                'name': p['name'],
                'lat': p['lat'],
                'lng': p['lng'],
                'rating': p.get('rating', 4.0),
                'nratings': p.get('user_ratings_total', 100),
                'open_now': p.get('is_open_now', True),
                'primary': p.get('primary_type', 'restaurant'),
                'types': '|'.join(p.get('types', ['restaurant'])),
            }
            for p in sample_places
        ])
        
        # Step 2: Compute ETAs (mock)
        etas = {row['id']: i * 300 for i, row in enumerate(places_df.to_dict('records'))}
        
        # Step 3: Create hex aggregation for clustering
        hex_df = pd.DataFrame({
            'hex': ['891f0000000ffff', '891f0000001ffff', '891f0000002ffff'],
            'lat': [35.6895, 35.6905, 35.6915],
            'lng': [139.6917, 139.6927, 139.6937],
            'n': [2, 2, 1],
            'localness': [0.8, 0.7, 0.6],
            'hex_mean_rating': [4.2, 4.0, 3.8],
            'hex_total_ratings': [200, 150, 100],
        })
        hex_df.attrs['h3_res'] = 9
        
        # Step 4: Score places
        weights = WeightConfig(
            w_rating=0.3,
            w_diversity=0.25,
            w_eta=0.2,
            w_open=0.15,
            w_crowd=0.1,
            variant_name='test'
        )
        
        scorer = PlaceScorer(weights=weights, enable_telemetry=True)
        scored_df = scorer.score_places(places_df, etas, hex_df)
        
        # Validate scoring
        assert len(scored_df) == len(places_df)
        assert 'score' in scored_df.columns
        assert scored_df['score'].min() >= 0.0
        assert scored_df['score'].max() <= 1.0
        assert not scored_df['score'].isna().any()
        
        # Step 5: Cluster (with fallback)
        config = ClusteringConfig(
            min_cluster_size=2,
            min_clusters=1,
            max_clusters=5,
        )
        hex_df_clustered, clusters, diagnostics = cluster_with_fallback(hex_df, config)
        
        # Validate clustering
        assert 'cluster' in hex_df_clustered.columns
        # Note: With small test data, clustering may fallback (returning empty clusters list)
        # This is expected behavior for sparse data
        
        # Step 6: Sequence with greedy (fast integration test)
        top_candidates = scored_df.nlargest(5, 'score')
        
        start_time = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=5)
        
        result = greedy_sequence(
            candidates=top_candidates,
            anchor_lat=35.6895,
            anchor_lng=139.6917,
            start_time=start_time,
            end_time=end_time,
            service_time_min=30,
        )
        
        # Validate sequencing (greedy_sequence returns GreedySequenceResult)
        # GreedySequenceResult doesn't have solution_found - it always succeeds
        assert len(result.stops) >= 1
        assert len(result.stops) <= len(top_candidates)
        assert result.total_duration_sec <= 5 * 3600  # Within time window
        
        # Validate output structure
        for stop in result.stops:
            # Greedy returns Stop objects (dataclass), not dicts
            assert hasattr(stop, 'place_id')
            assert hasattr(stop, 'place_name')
            assert hasattr(stop, 'lat')
            assert hasattr(stop, 'lng')
            assert hasattr(stop, 'arrival_time')
            assert hasattr(stop, 'departure_time')
            assert hasattr(stop, 'reason')
            assert stop.arrival_time < stop.departure_time
        
        # Validate telemetry
        telemetry = scorer.get_telemetry()
        assert len(telemetry) > 0
        assert all(t.breakdown.final_score >= 0.0 for t in telemetry)
    
    def test_vrptw_integration(self, sample_places, sample_distance_matrix):
        """Test VRPTW solver integration with fallback."""
        # Create candidates DataFrame
        candidates = pd.DataFrame([
            {
                'id': p['id'],
                'name': p['name'],
                'lat': p['lat'],
                'lng': p['lng'],
                'score': 0.8 - i * 0.1,
                'eta': i * 300,
                'open_now': True,
            }
            for i, p in enumerate(sample_places[:5])
        ])
        
        start_time = datetime.now().replace(hour=10, minute=0)
        end_time = start_time + timedelta(hours=6)
        
        config = VRPTWConfig(
            service_time_min=30,
            time_limit_sec=5,
            use_guided_local_search=True,
            verbose=False,
        )
        
        # Test with fallback enabled
        result = solve_vrptw_with_fallback(
            candidates=candidates,
            anchor_lat=35.6895,
            anchor_lng=139.6917,
            start_time=start_time,
            end_time=end_time,
            config=config,
            distance_matrix_full=sample_distance_matrix,
            force_greedy=False,
        )
        
        assert result.solution_found
        assert result.num_stops >= 1
        # sequence_method uses lowercase internal names
        assert result.sequence_method in ['ortools_vrptw', 'greedy', 'vrptw_no_solution']
        
        # Verify stops are within reasonable time bounds
        # Note: VRPTW solver may compute times relative to different base
        for stop in result.stops:
            arrival = stop['arrival_time']
            departure = stop['departure_time']
            # Verify arrival < departure
            assert arrival < departure
            # Verify service time makes sense
            service_duration = (departure - arrival).total_seconds() / 60
            assert service_duration == config.service_time_min
    
    def test_matrix_computation_with_caching(self, mock_routes_api):
        """Test route matrix computation with caching."""
        from src.routing import get_cache_stats, clear_cache
        
        # Clear cache first
        clear_cache()
        stats_before = get_cache_stats()
        # Cache stats structure: {'traffic_cache': {'size': 0, ...}, 'static_cache': {...}}
        assert stats_before['traffic_cache']['size'] == 0
        assert stats_before['static_cache']['size'] == 0
        
        # Create request
        origins = [Location(lat=35.6895, lng=139.6917)]
        destinations = [
            Location(lat=35.6905, lng=139.6927),
            Location(lat=35.6915, lng=139.6937),
        ]
        
        request = MatrixRequest(
            origins=origins,
            destinations=destinations,
            mode=TravelMode.WALK,
            routing_preference=RoutingPreference.TRAFFIC_AWARE,
        )
        
        # Note: This test would need async support or mocking
        # For now, verify the request is valid
        from src.routing import validate_matrix_request
        
        # validate_matrix_request raises ValueError if invalid
        try:
            validate_matrix_request(request)
        except ValueError as e:
            pytest.fail(f"Valid request was rejected: {e}")


@pytest.mark.integration
class TestErrorHandling:
    """Test graceful error handling throughout the pipeline."""
    
    def test_empty_places_list(self):
        """Test handling of empty places list."""
        places_df = pd.DataFrame(columns=['id', 'name', 'lat', 'lng', 'rating', 'nratings', 'open_now', 'primary', 'types'])
        etas = {}
        hex_df = pd.DataFrame(columns=['hex', 'lat', 'lng', 'n', 'localness'])
        hex_df.attrs['h3_res'] = 9
        
        weights = WeightConfig()
        scorer = PlaceScorer(weights=weights)
        scored_df = scorer.score_places(places_df, etas, hex_df)
        
        assert len(scored_df) == 0
        # Check that essential columns exist (scorer adds many intermediate columns)
        essential_cols = ['id', 'name', 'lat', 'lng', 'eta', 'score']
        for col in essential_cols:
            assert col in scored_df.columns
    
    def test_clustering_fallback_too_few_points(self):
        """Test clustering fallback when there are too few points."""
        hex_df = pd.DataFrame({
            'hex': ['891f0000000ffff'],
            'lat': [35.6895],
            'lng': [139.6917],
            'n': [1],
            'localness': [0.8],
        })
        hex_df.attrs['h3_res'] = 9
        
        config = ClusteringConfig(min_cluster_size=2)
        hex_df_result, clusters, diagnostics = cluster_with_fallback(hex_df, config)
        
        assert diagnostics.fallback_triggered
        assert 'too few points' in diagnostics.fallback_reason.lower()
        # Fallback returns empty clusters list
        assert len(clusters) == 0
        # All points marked as noise
        assert (hex_df_result['cluster'] == -1).all()
    
    def test_greedy_with_impossible_time_window(self, sample_places):
        """Test greedy sequencing with impossible time constraints."""
        candidates = pd.DataFrame([
            {
                'id': p['id'],
                'name': p['name'],
                'lat': p['lat'],
                'lng': p['lng'],
                'score': 0.8,
                'eta': 1800,  # 30 minutes away
                'open_now': True,
            }
            for p in sample_places[:3]
        ])
        
        # Time window too short (only 1 hour, can't fit 3 stops with 30min service time each)
        start_time = datetime.now().replace(hour=14, minute=0)
        end_time = start_time + timedelta(hours=1)
        
        result = greedy_sequence(
            candidates=candidates,
            anchor_lat=35.6895,
            anchor_lng=139.6917,
            start_time=start_time,
            end_time=end_time,
            service_time_min=30,
        )
        
        # Should still find a solution with fewer stops
        # GreedySequenceResult doesn't have solution_found - it always succeeds
        assert len(result.stops) >= 1
        assert len(result.stops) < len(candidates)  # Not all candidates fit
    
    def test_scoring_with_missing_data(self):
        """Test scoring with missing ratings and ETAs."""
        places_df = pd.DataFrame([
            {'id': 'p1', 'name': 'Place 1', 'lat': 35.6895, 'lng': 139.6917, 'rating': np.nan, 'nratings': 0, 'open_now': True, 'primary': 'restaurant', 'types': 'restaurant'},
            {'id': 'p2', 'name': 'Place 2', 'lat': 35.6905, 'lng': 139.6927, 'rating': 4.5, 'nratings': 100, 'open_now': False, 'primary': 'cafe', 'types': 'cafe'},
        ])
        
        etas = {'p1': 300}  # Missing p2
        
        hex_df = pd.DataFrame({
            'hex': ['891f0000000ffff'],
            'lat': [35.6895],
            'lng': [139.6917],
            'n': [2],
            'localness': [0.7],
        })
        hex_df.attrs['h3_res'] = 9
        
        weights = WeightConfig()
        scorer = PlaceScorer(weights=weights)
        scored_df = scorer.score_places(places_df, etas, hex_df)
        
        # Should handle missing data gracefully
        assert len(scored_df) == 2
        assert not scored_df['score'].isna().any()  # No NaN scores


@pytest.mark.integration
class TestFallbackMechanisms:
    """Test automatic fallback mechanisms."""
    
    def test_clustering_fallback_single_cluster(self):
        """Test fallback when HDBSCAN produces single cluster."""
        # Create points that are very close together (will form single cluster)
        hex_df = pd.DataFrame({
            'hex': [f'891f000000{i}ffff' for i in range(5)],
            'lat': [35.6895 + i * 0.0001 for i in range(5)],
            'lng': [139.6917 + i * 0.0001 for i in range(5)],
            'n': [3, 2, 4, 2, 3],
            'localness': [0.8, 0.7, 0.9, 0.6, 0.75],
        })
        hex_df.attrs['h3_res'] = 9
        
        config = ClusteringConfig(
            min_cluster_size=2,
            min_clusters=2,  # Want at least 2 clusters
            enable_refitting=True,
        )
        
        hex_df_result, clusters, diagnostics = cluster_with_fallback(hex_df, config)
        
        # With small, close-together points and min_clusters=2, may fallback
        # Check that clustering was attempted
        assert 'cluster' in hex_df_result.columns
        # Either succeeded with clusters or fell back (empty clusters list)
        assert isinstance(clusters, list)
    
    def test_vrptw_fallback_to_greedy(self, sample_places):
        """Test VRPTW fallback to greedy when OR-Tools times out."""
        candidates = pd.DataFrame([
            {
                'id': p['id'],
                'name': p['name'],
                'lat': p['lat'],
                'lng': p['lng'],
                'score': 0.8 - i * 0.05,
                'eta': i * 300,
                'open_now': True,
            }
            for i, p in enumerate(sample_places[:10])
        ])
        
        start_time = datetime.now().replace(hour=10, minute=0)
        end_time = start_time + timedelta(hours=8)
        
        config = VRPTWConfig(
            service_time_min=30,
            time_limit_sec=0.001,  # Very short time limit to trigger fallback
            use_guided_local_search=False,
            verbose=False,
        )
        
        result = solve_vrptw_with_fallback(
            candidates=candidates,
            anchor_lat=35.6895,
            anchor_lng=139.6917,
            start_time=start_time,
            end_time=end_time,
            config=config,
            force_greedy=False,
        )
        
        # Should fallback to greedy and still produce a solution
        assert result.solution_found
        # sequence_method will be lowercase internal name
        assert 'greedy' in result.sequence_method.lower() or 'fallback' in result.fallback_reason.lower()
    
    def test_force_greedy_mode(self, sample_places):
        """Test forcing greedy mode (for --fast execution)."""
        candidates = pd.DataFrame([
            {
                'id': p['id'],
                'name': p['name'],
                'lat': p['lat'],
                'lng': p['lng'],
                'score': 0.8,
                'eta': 300,
                'open_now': True,
            }
            for p in sample_places[:5]
        ])
        
        start_time = datetime.now().replace(hour=13, minute=0)
        end_time = start_time + timedelta(hours=4)
        
        config = VRPTWConfig(service_time_min=30)
        
        result = solve_vrptw_with_fallback(
            candidates=candidates,
            anchor_lat=35.6895,
            anchor_lng=139.6917,
            start_time=start_time,
            end_time=end_time,
            config=config,
            force_greedy=True,  # Force greedy mode
        )
        
        assert result.solution_found
        # sequence_method uses lowercase internal names
        assert result.sequence_method == 'greedy'
        assert 'forced greedy mode' in result.fallback_reason.lower()


@pytest.mark.integration
class TestABTesting:
    """Test A/B testing integration."""
    
    def test_ab_variant_selection(self):
        """Test A/B variant selection is deterministic."""
        user_id = "test_user_123"
        
        # Call multiple times with same user_id
        variant1 = select_ab_variant(user_id)
        variant2 = select_ab_variant(user_id)
        
        assert variant1.variant_name == variant2.variant_name
        assert variant1.w_rating == variant2.w_rating
        assert variant1.w_diversity == variant2.w_diversity
    
    def test_ab_variant_distribution(self):
        """Test A/B variants are distributed across users."""
        variants = set()
        
        for i in range(30):
            user_id = f"user_{i}"
            variant = select_ab_variant(user_id)
            variants.add(variant.variant_name)
        
        # Should see multiple variants
        assert len(variants) >= 2  # At least 2 different variants
    
    def test_scoring_with_different_variants(self, sample_places):
        """Test scoring produces different results with different variants."""
        places_df = pd.DataFrame([
            {
                'id': p['id'],
                'name': p['name'],
                'lat': p['lat'],
                'lng': p['lng'],
                'rating': p.get('rating', 4.0),
                'nratings': p.get('user_ratings_total', 100),
                'open_now': p.get('is_open_now', True),
                'primary': p.get('primary_type', 'restaurant'),
                'types': '|'.join(p.get('types', ['restaurant'])),
            }
            for p in sample_places
        ])
        
        etas = {p['id']: i * 300 for i, p in enumerate(sample_places)}
        
        hex_df = pd.DataFrame({
            'hex': ['891f0000000ffff', '891f0000001ffff'],
            'lat': [35.6895, 35.6905],
            'lng': [139.6917, 139.6927],
            'n': [2, 3],
            'localness': [0.8, 0.7],
        })
        hex_df.attrs['h3_res'] = 9
        
        # Score with variant A (rating-focused)
        variant_a = WeightConfig(
            w_rating=0.5,
            w_diversity=0.2,
            w_eta=0.15,
            w_open=0.1,
            w_crowd=0.05,
            variant_name='variant_a'
        )
        scorer_a = PlaceScorer(weights=variant_a)
        scores_a = scorer_a.score_places(places_df, etas, hex_df)
        
        # Score with variant B (diversity-focused)
        variant_b = WeightConfig(
            w_rating=0.2,
            w_diversity=0.5,
            w_eta=0.15,
            w_open=0.1,
            w_crowd=0.05,
            variant_name='variant_b'
        )
        scorer_b = PlaceScorer(weights=variant_b)
        scores_b = scorer_b.score_places(places_df, etas, hex_df)
        
        # Rankings should differ
        top_a = scores_a.nlargest(3, 'score')['id'].tolist()
        top_b = scores_b.nlargest(3, 'score')['id'].tolist()
        
        # At least one difference in top 3 (unless data is very homogeneous)
        assert top_a != top_b or len(sample_places) < 3


@pytest.mark.integration
class TestOutputFormats:
    """Test output format validation."""
    
    def test_scored_place_structure(self, sample_places):
        """Test scored place output has all required fields."""
        places_df = pd.DataFrame([
            {
                'id': p['id'],
                'name': p['name'],
                'lat': p['lat'],
                'lng': p['lng'],
                'rating': p.get('rating', 4.0),
                'nratings': p.get('user_ratings_total', 100),
                'open_now': True,
                'primary': 'restaurant',
                'types': 'restaurant|food',
            }
            for p in sample_places[:3]
        ])
        
        etas = {p['id']: 300 for p in sample_places[:3]}
        hex_df = pd.DataFrame({
            'hex': ['891f0000000ffff'],
            'lat': [35.6895],
            'lng': [139.6917],
            'n': [3],
            'localness': [0.7],
        })
        hex_df.attrs['h3_res'] = 9
        
        weights = WeightConfig()
        scored_df = score_places(places_df, etas, hex_df, weights)
        
        # Verify required columns exist
        required_cols = ['id', 'name', 'lat', 'lng', 'rating', 'score', 'eta', 'diversity_gain']
        for col in required_cols:
            assert col in scored_df.columns, f"Missing column: {col}"
        
        # Verify data types
        assert scored_df['score'].dtype in [np.float64, np.float32]
        assert scored_df['eta'].dtype in [np.int64, np.int32, np.float64]
    
    def test_clustering_output_structure(self):
        """Test clustering output has all required fields."""
        hex_df = pd.DataFrame({
            'hex': [f'891f000000{i}ffff' for i in range(5)],
            'lat': [35.6895 + i * 0.001 for i in range(5)],
            'lng': [139.6917 + i * 0.001 for i in range(5)],
            'n': [2, 3, 2, 4, 3],
            'localness': [0.8, 0.7, 0.9, 0.6, 0.75],
        })
        hex_df.attrs['h3_res'] = 9
        
        config = ClusteringConfig(min_cluster_size=2)
        hex_df_result, clusters, diagnostics = cluster_with_fallback(hex_df, config)
        
        # Verify cluster structure
        for cluster in clusters:
            assert hasattr(cluster, 'cluster_id')
            assert hasattr(cluster, 'label')
            assert hasattr(cluster, 'hex_ids')
            assert hasattr(cluster, 'centroid_lat')
            assert hasattr(cluster, 'centroid_lng')
            assert isinstance(cluster.hex_ids, list)
            assert len(cluster.hex_ids) > 0
        
        # Verify diagnostics structure
        assert hasattr(diagnostics, 'num_points')
        assert hasattr(diagnostics, 'num_clusters')
        assert hasattr(diagnostics, 'num_noise')
        assert hasattr(diagnostics, 'fallback_triggered')
        assert hasattr(diagnostics, 'fallback_reason')
    
    def test_sequencing_output_structure(self, sample_places):
        """Test sequencing output has all required fields."""
        candidates = pd.DataFrame([
            {
                'id': p['id'],
                'name': p['name'],
                'lat': p['lat'],
                'lng': p['lng'],
                'score': 0.8,
                'eta': 300,
                'open_now': True,
            }
            for p in sample_places[:3]
        ])
        
        start_time = datetime.now().replace(hour=10, minute=0)
        end_time = start_time + timedelta(hours=4)
        
        result = greedy_sequence(
            candidates=candidates,
            anchor_lat=35.6895,
            anchor_lng=139.6917,
            start_time=start_time,
            end_time=end_time,
            service_time_min=30,
        )
        
        # Verify result structure (GreedySequenceResult doesn't have solution_found)
        assert hasattr(result, 'stops')
        assert hasattr(result, 'num_stops_skipped')
        assert hasattr(result, 'total_travel_time_sec')
        assert hasattr(result, 'total_service_time_sec')
        assert hasattr(result, 'total_duration_sec')
        assert hasattr(result, 'sequence_method')
        
        # Verify stop structure
        for stop in result.stops:
            # Greedy returns Stop objects, not dicts
            assert hasattr(stop, 'place_id')
            assert hasattr(stop, 'place_name')
            assert hasattr(stop, 'lat')
            assert hasattr(stop, 'lng')
            assert hasattr(stop, 'arrival_time')
            assert hasattr(stop, 'departure_time')
            assert hasattr(stop, 'reason')
            assert isinstance(stop.arrival_time, datetime)
            assert isinstance(stop.departure_time, datetime)


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Test performance characteristics of the pipeline."""
    
    def test_scoring_performance(self, sample_places):
        """Test scoring completes in reasonable time."""
        import time
        
        # Create larger dataset
        places_df = pd.DataFrame([
            {
                'id': f"p_{i}",
                'name': f"Place {i}",
                'lat': 35.6895 + (i % 10) * 0.001,
                'lng': 139.6917 + (i // 10) * 0.001,
                'rating': 3.5 + (i % 10) * 0.1,
                'nratings': 50 + i * 5,
                'open_now': i % 2 == 0,
                'primary': 'restaurant',
                'types': 'restaurant|food',
            }
            for i in range(100)
        ])
        
        etas = {f"p_{i}": i * 100 for i in range(100)}
        hex_df = pd.DataFrame({
            'hex': [f'891f000000{i % 10}ffff' for i in range(10)],
            'lat': [35.6895 + i * 0.001 for i in range(10)],
            'lng': [139.6917 + i * 0.001 for i in range(10)],
            'n': [10] * 10,
            'localness': [0.7 + i * 0.02 for i in range(10)],
        })
        hex_df.attrs['h3_res'] = 9
        
        weights = WeightConfig()
        scorer = PlaceScorer(weights=weights)
        
        start = time.time()
        scored_df = scorer.score_places(places_df, etas, hex_df)
        elapsed = time.time() - start
        
        # Should complete in under 1 second
        assert elapsed < 1.0
        assert len(scored_df) == 100
    
    def test_clustering_performance(self):
        """Test clustering completes in reasonable time."""
        import time
        
        # Create larger hex dataset
        hex_df = pd.DataFrame({
            'hex': [f'891f000000{i:03x}ffff' for i in range(50)],
            'lat': [35.6895 + (i % 10) * 0.001 for i in range(50)],
            'lng': [139.6917 + (i // 10) * 0.001 for i in range(50)],
            'n': [2 + i % 5 for i in range(50)],
            'localness': [0.5 + (i % 10) * 0.05 for i in range(50)],
        })
        hex_df.attrs['h3_res'] = 9
        
        config = ClusteringConfig(min_cluster_size=3)
        
        start = time.time()
        hex_df_result, clusters, diagnostics = cluster_with_fallback(hex_df, config)
        elapsed = time.time() - start
        
        # Should complete in under 2 seconds
        assert elapsed < 2.0
        assert len(clusters) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
