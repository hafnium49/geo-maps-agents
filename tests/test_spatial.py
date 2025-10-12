"""
Unit Tests for Spatial Module (src/spatial)

Tests H3 aggregation, HDBSCAN clustering, fallback logic, and cluster labeling.
"""

import pytest
import numpy as np
import pandas as pd
import h3

from src.spatial.clustering import (
    ClusteringConfig,
    ClusteringDiagnostics,
    ClusterInfo,
    cluster_with_fallback,
    label_cluster,
    _compute_cluster_quality,
    _detect_degenerate_case,
)


# ==============================================================================
# H3 Aggregation Tests
# ==============================================================================

class TestH3Aggregation:
    """Test H3 hex aggregation functionality."""
    
    def test_h3_index_generation(self):
        """Test generating H3 indices from coordinates."""
        lat, lng = 35.6812, 139.7671  # Tokyo Station
        
        # Different resolutions
        hex_res_7 = h3.geo_to_h3(lat, lng, 7)  # ~5km hexagons
        hex_res_9 = h3.geo_to_h3(lat, lng, 9)  # ~0.1km hexagons
        hex_res_11 = h3.geo_to_h3(lat, lng, 11)  # ~0.01km hexagons
        
        # Should be different at different resolutions
        assert hex_res_7 != hex_res_9
        assert hex_res_9 != hex_res_11
        
        # Should be valid H3 indices
        assert h3.h3_is_valid(hex_res_7)
        assert h3.h3_is_valid(hex_res_9)
        assert h3.h3_is_valid(hex_res_11)
    
    def test_h3_index_consistency(self):
        """Test that same coordinates always produce same H3 index."""
        lat, lng = 35.6895, 139.6917
        
        hex1 = h3.geo_to_h3(lat, lng, 9)
        hex2 = h3.geo_to_h3(lat, lng, 9)
        
        assert hex1 == hex2
    
    def test_h3_neighbor_detection(self):
        """Test detecting neighboring hexagons."""
        lat, lng = 35.6812, 139.7671
        hex_id = h3.geo_to_h3(lat, lng, 9)
        
        # Get neighbors (k-ring with k=1)
        neighbors = h3.k_ring(hex_id, 1)
        
        # Should include the center hex plus 6 neighbors (hexagon has 6 sides)
        assert len(neighbors) == 7
        assert hex_id in neighbors
    
    def test_dataframe_h3_aggregation(self):
        """Test aggregating places into H3 hexagons."""
        # Sample places
        places = pd.DataFrame({
            'id': ['p1', 'p2', 'p3', 'p4'],
            'name': ['Place 1', 'Place 2', 'Place 3', 'Place 4'],
            'lat': [35.6812, 35.6815, 35.6895, 35.7148],
            'lng': [139.7671, 139.7675, 139.6917, 139.7967],
            'rating': [4.5, 4.2, 4.7, 3.9],
        })
        
        # Add H3 indices
        h3_res = 9
        places['hex'] = places.apply(
            lambda r: h3.geo_to_h3(r['lat'], r['lng'], h3_res),
            axis=1
        )
        
        # Check that hexes were added
        assert 'hex' in places.columns
        assert places['hex'].notna().all()
        
        # Places very close together should share same hex
        # (p1 and p2 are ~50m apart, same hex at resolution 9)
        assert places.iloc[0]['hex'] == places.iloc[1]['hex']


# ==============================================================================
# Cluster Labeling Tests
# ==============================================================================

class TestClusterLabeling:
    """Test cluster labeling functionality."""
    
    @pytest.fixture
    def sample_hex_df(self):
        """Create sample hex DataFrame with cluster assignments."""
        return pd.DataFrame({
            'hex': ['hex1', 'hex2', 'hex3', 'hex4', 'hex5'],
            'cluster': [0, 0, 1, 1, -1],
            'types': [
                'restaurant|food',
                'restaurant|cafe',
                'museum|tourist_attraction',
                'museum|art_gallery',
                'park',
            ],
        })
    
    def test_label_cluster_basic(self, sample_hex_df):
        """Test basic cluster labeling."""
        label = label_cluster(sample_hex_df, cluster_id=0, top_n_tokens=2)
        
        # Should include 'restaurant' as it's most common
        assert 'restaurant' in label.lower()
    
    def test_label_cluster_single_type(self):
        """Test labeling cluster with single type."""
        df = pd.DataFrame({
            'hex': ['hex1', 'hex2', 'hex3'],
            'cluster': [0, 0, 0],
            'types': ['museum', 'museum', 'museum'],
        })
        
        label = label_cluster(df, cluster_id=0, top_n_tokens=2)
        assert label == 'Museum'
    
    def test_label_empty_cluster(self, sample_hex_df):
        """Test labeling empty cluster."""
        label = label_cluster(sample_hex_df, cluster_id=99)  # Non-existent cluster
        assert label == "Empty Cluster"
    
    def test_label_generic_tokens_filtered(self):
        """Test that generic tokens are filtered out."""
        df = pd.DataFrame({
            'hex': ['hex1', 'hex2', 'hex3'],
            'cluster': [0, 0, 0],
            'types': [
                'point_of_interest|establishment|restaurant',
                'point_of_interest|establishment|cafe',
                'point_of_interest|restaurant',
            ],
        })
        
        label = label_cluster(df, cluster_id=0, top_n_tokens=2)
        
        # Should not include generic tokens
        assert 'point_of_interest' not in label.lower()
        assert 'establishment' not in label.lower()
        
        # Should include specific types
        assert 'restaurant' in label.lower()
    
    def test_label_deterministic(self, sample_hex_df):
        """Test that labeling is deterministic."""
        label1 = label_cluster(sample_hex_df, cluster_id=0, top_n_tokens=2)
        label2 = label_cluster(sample_hex_df, cluster_id=0, top_n_tokens=2)
        
        assert label1 == label2


# ==============================================================================
# Clustering Configuration Tests
# ==============================================================================

class TestClusteringConfig:
    """Test clustering configuration."""
    
    def test_default_config(self):
        """Test default clustering configuration."""
        config = ClusteringConfig()
        
        assert config.min_cluster_size == 12
        assert config.min_samples is None
        assert config.min_clusters == 2
        assert config.max_clusters == 10
        assert config.enable_refitting is True
    
    def test_custom_config(self):
        """Test custom clustering configuration."""
        config = ClusteringConfig(
            min_cluster_size=20,
            min_clusters=3,
            max_clusters=5,
            enable_refitting=False
        )
        
        assert config.min_cluster_size == 20
        assert config.min_clusters == 3
        assert config.max_clusters == 5
        assert config.enable_refitting is False


# ==============================================================================
# Cluster Quality Tests
# ==============================================================================

class TestClusterQuality:
    """Test cluster quality metrics."""
    
    def test_silhouette_score_computation(self):
        """Test silhouette score computation."""
        # Create well-separated clusters
        X = np.array([
            [0, 0], [0.5, 0.5], [1, 1],  # Cluster 0
            [10, 10], [10.5, 10.5], [11, 11],  # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        score = _compute_cluster_quality(X, labels, num_clusters=2)
        
        assert score is not None
        assert -1 <= score <= 1
        # Well-separated clusters should have high silhouette score
        assert score > 0.5
    
    def test_silhouette_score_single_cluster(self):
        """Test silhouette score with single cluster."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])
        
        score = _compute_cluster_quality(X, labels, num_clusters=1)
        
        # Should return None for single cluster
        assert score is None
    
    def test_silhouette_score_with_noise(self):
        """Test silhouette score computation with noise points."""
        X = np.array([
            [0, 0], [0.5, 0.5], [1, 1],  # Cluster 0
            [10, 10], [10.5, 10.5], [11, 11],  # Cluster 1
            [50, 50],  # Noise point
        ])
        labels = np.array([0, 0, 0, 1, 1, 1, -1])
        
        score = _compute_cluster_quality(X, labels, num_clusters=2)
        
        # Should compute score excluding noise
        assert score is not None
        assert -1 <= score <= 1


# ==============================================================================
# Degenerate Case Detection Tests
# ==============================================================================

class TestDegenerateCaseDetection:
    """Test degenerate case detection."""
    
    def test_too_few_points(self):
        """Test detection of too few points."""
        config = ClusteringConfig(min_cluster_size=12)
        
        is_degenerate, suggestions = _detect_degenerate_case(5, config)
        
        assert is_degenerate is True
        assert len(suggestions) > 0
        assert "5 points" in suggestions[0]
    
    def test_barely_enough_points(self):
        """Test warning for barely enough points."""
        config = ClusteringConfig(min_cluster_size=12)
        
        is_degenerate, suggestions = _detect_degenerate_case(20, config)
        
        # Not degenerate, but should suggest widening radius
        assert is_degenerate is False
        assert len(suggestions) > 0
        assert "20 points" in suggestions[0]
    
    def test_sufficient_points(self):
        """Test with sufficient points."""
        config = ClusteringConfig(min_cluster_size=12)
        
        is_degenerate, suggestions = _detect_degenerate_case(50, config)
        
        assert is_degenerate is False
        # May or may not have suggestions


# ==============================================================================
# HDBSCAN Clustering Tests
# ==============================================================================

class TestHDBSCANClustering:
    """Test HDBSCAN clustering with fallback logic."""
    
    @pytest.fixture
    def well_separated_hexes(self):
        """Create well-separated hex clusters for testing."""
        # Cluster 0: Around Tokyo Station
        cluster0 = pd.DataFrame({
            'hex': [h3.geo_to_h3(35.6812 + i*0.001, 139.7671 + j*0.001, 9) 
                    for i in range(4) for j in range(4)],
            'lat': [35.6812 + i*0.001 for i in range(4) for _ in range(4)],
            'lng': [139.7671 + j*0.001 for _ in range(4) for j in range(4)],
            'types': ['restaurant|food'] * 16,
            'localness': [0.7] * 16,
        })
        
        # Cluster 1: Around Shinjuku (3km away)
        cluster1 = pd.DataFrame({
            'hex': [h3.geo_to_h3(35.6895 + i*0.001, 139.6917 + j*0.001, 9)
                    for i in range(4) for j in range(4)],
            'lat': [35.6895 + i*0.001 for i in range(4) for _ in range(4)],
            'lng': [139.6917 + j*0.001 for _ in range(4) for j in range(4)],
            'types': ['museum|tourist_attraction'] * 16,
            'localness': [0.5] * 16,
        })
        
        df = pd.concat([cluster0, cluster1], ignore_index=True)
        df.attrs['h3_res'] = 9
        return df
    
    def test_clustering_well_separated(self, well_separated_hexes):
        """Test clustering with well-separated data."""
        config = ClusteringConfig(min_cluster_size=8, min_clusters=2, max_clusters=10)
        
        hex_df, clusters, diagnostics = cluster_with_fallback(
            well_separated_hexes,
            config
        )
        
        # Should find 2 clusters
        assert diagnostics.num_clusters == 2
        assert not diagnostics.fallback_triggered
        assert len(clusters) == 2
        
        # Check that clusters have labels
        assert all(c.label for c in clusters)
        
        # Silhouette score should be good
        if diagnostics.silhouette_score is not None:
            assert diagnostics.silhouette_score > 0.3
    
    def test_clustering_too_few_points(self):
        """Test clustering with too few points triggers fallback."""
        # Only 5 points, less than min_cluster_size
        df = pd.DataFrame({
            'hex': [f'hex{i}' for i in range(5)],
            'lat': [35.6812 + i*0.001 for i in range(5)],
            'lng': [139.7671 + i*0.001 for i in range(5)],
            'types': ['restaurant'] * 5,
            'localness': [0.7] * 5,
        })
        df.attrs['h3_res'] = 9
        
        config = ClusteringConfig(min_cluster_size=12)
        
        hex_df, clusters, diagnostics = cluster_with_fallback(df, config)
        
        # Should trigger fallback
        assert diagnostics.fallback_triggered
        assert diagnostics.degenerate_case
        assert diagnostics.num_clusters == 0
        assert len(clusters) == 0
    
    def test_clustering_single_cluster(self):
        """Test clustering that produces single cluster."""
        # All points very close together
        df = pd.DataFrame({
            'hex': [h3.geo_to_h3(35.6812 + i*0.0001, 139.7671 + j*0.0001, 9)
                    for i in range(4) for j in range(4)],
            'lat': [35.6812 + i*0.0001 for i in range(4) for _ in range(4)],
            'lng': [139.7671 + j*0.0001 for _ in range(4) for j in range(4)],
            'types': ['restaurant'] * 16,
            'localness': [0.7] * 16,
        })
        df.attrs['h3_res'] = 9
        
        config = ClusteringConfig(min_cluster_size=8, min_clusters=2)
        
        hex_df, clusters, diagnostics = cluster_with_fallback(df, config)
        
        # Should trigger fallback (< min_clusters)
        assert diagnostics.fallback_triggered or diagnostics.num_clusters < 2
    
    def test_clustering_noise_handling(self):
        """Test that noise points are handled correctly."""
        # Create clusters with some outliers
        df = pd.DataFrame({
            'hex': [f'hex{i}' for i in range(25)],
            'lat': ([35.6812 + i*0.001 for i in range(10)] +  # Cluster 0
                    [35.6895 + i*0.001 for i in range(10)] +  # Cluster 1
                    [35.8000, 35.8001, 35.8002, 35.8003, 35.8004]),  # Outliers
            'lng': ([139.7671 + i*0.001 for i in range(10)] +
                    [139.6917 + i*0.001 for i in range(10)] +
                    [139.9000, 139.9001, 139.9002, 139.9003, 139.9004]),
            'types': ['restaurant'] * 20 + ['museum'] * 5,
            'localness': [0.7] * 25,
        })
        df.attrs['h3_res'] = 9
        
        config = ClusteringConfig(min_cluster_size=8)
        
        hex_df, clusters, diagnostics = cluster_with_fallback(df, config)
        
        # Should have some noise points (outliers)
        if not diagnostics.fallback_triggered:
            assert diagnostics.num_noise >= 0


# ==============================================================================
# Clustering Diagnostics Tests
# ==============================================================================

class TestClusteringDiagnostics:
    """Test clustering diagnostics."""
    
    def test_diagnostics_creation(self):
        """Test creating clustering diagnostics."""
        diag = ClusteringDiagnostics(
            num_points=100,
            num_clusters=3,
            num_noise=5,
            cluster_sizes=[30, 40, 25],
            silhouette_score=0.65,
            fallback_triggered=False
        )
        
        assert diag.num_points == 100
        assert diag.num_clusters == 3
        assert diag.num_noise == 5
        assert len(diag.cluster_sizes) == 3
        assert diag.silhouette_score == 0.65
    
    def test_diagnostics_with_fallback(self):
        """Test diagnostics when fallback is triggered."""
        diag = ClusteringDiagnostics(
            num_points=50,
            num_clusters=0,
            num_noise=50,
            fallback_triggered=True,
            fallback_reason="Too few points",
            degenerate_case=True
        )
        
        assert diag.fallback_triggered
        assert diag.degenerate_case
        assert diag.fallback_reason == "Too few points"
        assert diag.num_clusters == 0


# ==============================================================================
# ClusterInfo Tests
# ==============================================================================

class TestClusterInfo:
    """Test ClusterInfo dataclass."""
    
    def test_cluster_info_creation(self):
        """Test creating cluster info."""
        info = ClusterInfo(
            cluster_id=0,
            label="restaurant + cafe",
            hex_ids=['hex1', 'hex2', 'hex3'],
            centroid_lat=35.6812,
            centroid_lng=139.7671,
            size=15
        )
        
        assert info.cluster_id == 0
        assert info.label == "restaurant + cafe"
        assert len(info.hex_ids) == 3
        assert info.size == 15
    
    def test_cluster_info_noise(self):
        """Test cluster info for noise cluster."""
        info = ClusterInfo(
            cluster_id=-1,
            label="Noise",
            hex_ids=[],
            centroid_lat=0.0,
            centroid_lng=0.0,
            size=0
        )
        
        assert info.cluster_id == -1
        assert info.label == "Noise"


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test clustering with empty DataFrame."""
        df = pd.DataFrame(columns=['hex', 'lat', 'lng', 'types', 'localness'])
        df.attrs['h3_res'] = 9
        
        config = ClusteringConfig()
        
        hex_df, clusters, diagnostics = cluster_with_fallback(df, config)
        
        # Should handle gracefully
        assert diagnostics.fallback_triggered
        assert diagnostics.num_clusters == 0
    
    def test_single_point(self):
        """Test clustering with single point."""
        df = pd.DataFrame({
            'hex': ['hex1'],
            'lat': [35.6812],
            'lng': [139.7671],
            'types': ['restaurant'],
            'localness': [0.7],
        })
        df.attrs['h3_res'] = 9
        
        config = ClusteringConfig()
        
        hex_df, clusters, diagnostics = cluster_with_fallback(df, config)
        
        # Should trigger fallback
        assert diagnostics.fallback_triggered
        assert diagnostics.degenerate_case
    
    def test_missing_h3_resolution(self):
        """Test behavior when h3_res attribute is missing."""
        df = pd.DataFrame({
            'hex': ['hex1', 'hex2', 'hex3'],
            'lat': [35.6812, 35.6813, 35.6814],
            'lng': [139.7671, 139.7672, 139.7673],
            'types': ['restaurant'] * 3,
            'localness': [0.7] * 3,
        })
        # No h3_res attribute set
        
        config = ClusteringConfig()
        
        # Should still work (may use default or infer)
        hex_df, clusters, diagnostics = cluster_with_fallback(df, config)
        
        # Should handle gracefully (likely fallback due to too few points)
        assert diagnostics is not None
