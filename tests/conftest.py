"""
Pytest configuration and shared fixtures for geo-maps-agents tests.

This file provides:
- Mock API responses (Places API, Routes API)
- Test data fixtures
- Common test utilities
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np
import h3


# ==============================================================================
# Test Data Paths
# ==============================================================================

@pytest.fixture
def fixtures_dir() -> Path:
    """Path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def places_fixture(fixtures_dir) -> Dict[str, Any]:
    """Load Places API mock responses."""
    fixture_file = fixtures_dir / "places_api.json"
    if fixture_file.exists():
        with open(fixture_file) as f:
            return json.load(f)
    return {}


@pytest.fixture
def routes_fixture(fixtures_dir) -> Dict[str, Any]:
    """Load Routes API mock responses."""
    fixture_file = fixtures_dir / "routes_api.json"
    if fixture_file.exists():
        with open(fixture_file) as f:
            return json.load(f)
    return {}


# ==============================================================================
# Sample Places Data
# ==============================================================================

@pytest.fixture
def sample_places() -> List[Dict[str, Any]]:
    """Sample places data for testing."""
    return [
        {
            "id": "place_1",
            "name": "Tokyo Station",
            "lat": 35.6812,
            "lng": 139.7671,
            "rating": 4.5,
            "user_ratings_total": 50000,
            "types": ["train_station", "transit_station"],
            "price_level": 0,
        },
        {
            "id": "place_2",
            "name": "Senso-ji Temple",
            "lat": 35.7148,
            "lng": 139.7967,
            "rating": 4.4,
            "user_ratings_total": 35000,
            "types": ["place_of_worship", "tourist_attraction"],
            "price_level": 0,
        },
        {
            "id": "place_3",
            "name": "Shibuya Crossing",
            "lat": 35.6595,
            "lng": 139.7004,
            "rating": 4.3,
            "user_ratings_total": 25000,
            "types": ["tourist_attraction", "point_of_interest"],
            "price_level": 0,
        },
        {
            "id": "place_4",
            "name": "Meiji Shrine",
            "lat": 35.6764,
            "lng": 139.6993,
            "rating": 4.6,
            "user_ratings_total": 40000,
            "types": ["place_of_worship", "tourist_attraction"],
            "price_level": 0,
        },
        {
            "id": "place_5",
            "name": "Tokyo Skytree",
            "lat": 35.7101,
            "lng": 139.8107,
            "rating": 4.5,
            "user_ratings_total": 60000,
            "types": ["tourist_attraction", "point_of_interest"],
            "price_level": 3,
        },
    ]


@pytest.fixture
def sample_places_df(sample_places) -> pd.DataFrame:
    """Sample places as DataFrame."""
    return pd.DataFrame(sample_places)


@pytest.fixture
def sample_places_for_scoring() -> pd.DataFrame:
    """Sample places DataFrame with all fields needed for scoring."""
    return pd.DataFrame({
        'id': ['p1', 'p2', 'p3', 'p4'],
        'name': ['Place 1', 'Place 2', 'Place 3', 'Place 4'],
        'lat': [35.6812, 35.6895, 35.6764, 35.7148],
        'lng': [139.7671, 139.6917, 139.6993, 139.7967],
        'rating': [4.5, 4.2, 4.7, 3.9],
        'nratings': [1500, 800, 2100, 600],
        'open_now': [1, 1, 0, 1],
        'primary': ['restaurant', 'cafe', 'restaurant', 'museum'],
        'types': ['restaurant|food', 'cafe|restaurant', 'restaurant|sushi', 'museum|tourist_attraction'],
    })


# ==============================================================================
# Sample Distance Matrix
# ==============================================================================

@pytest.fixture
def sample_distance_matrix() -> np.ndarray:
    """Sample distance matrix (5x5, in seconds)."""
    return np.array([
        [0, 600, 1200, 900, 1500],
        [600, 0, 800, 700, 1000],
        [1200, 800, 0, 300, 1800],
        [900, 700, 300, 0, 1600],
        [1500, 1000, 1800, 1600, 0],
    ], dtype=np.int32)


# ==============================================================================
# Sample Scored Places
# ==============================================================================

@pytest.fixture
def sample_scored_places() -> pd.DataFrame:
    """Sample scored places for routing tests."""
    return pd.DataFrame({
        'id': ['p1', 'p2', 'p3', 'p4'],
        'name': ['Place 1', 'Place 2', 'Place 3', 'Place 4'],
        'lat': [35.6895, 35.6905, 35.6915, 35.6925],
        'lng': [139.6917, 139.6927, 139.6937, 139.6947],
        'score': [0.9, 0.8, 0.7, 0.6],
        'eta': [300, 600, 900, 1200],  # seconds from anchor
    })


@pytest.fixture
def sample_etas():
    """Sample ETAs dict for testing."""
    return {
        'p1': 300,
        'p2': 600,
        'p3': 900,
        'p4': 450,
    }


@pytest.fixture
def sample_hex_df():
    """Sample hex-level aggregated features."""
    # Generate H3 indices for sample locations
    hexes = [
        h3.geo_to_h3(35.6812, 139.7671, 9),
        h3.geo_to_h3(35.6895, 139.6917, 9),
        h3.geo_to_h3(35.6764, 139.6993, 9),
        h3.geo_to_h3(35.7148, 139.7967, 9),
    ]
    
    df = pd.DataFrame({
        'hex': hexes,
        'localness': [0.7, 0.5, 0.8, 0.6],
        'cluster': [0, 1, 0, 2],
    })
    df.attrs['h3_res'] = 9
    return df


# ==============================================================================
# Test Time Windows
# ==============================================================================

@pytest.fixture
def test_start_time() -> datetime:
    """Test tour start time."""
    return datetime(2025, 10, 12, 13, 0, 0)


@pytest.fixture
def test_end_time(test_start_time) -> datetime:
    """Test tour end time (4 hours later)."""
    return test_start_time + timedelta(hours=4)


# ==============================================================================
# Mock Functions
# ==============================================================================

class MockPlacesAPI:
    """Mock Places API for testing."""
    
    def __init__(self, places_data: List[Dict[str, Any]]):
        self.places_data = places_data
    
    def search_nearby(self, lat: float, lng: float, radius: int = 5000, **kwargs):
        """Mock nearby search."""
        return {"results": self.places_data[:3], "status": "OK"}
    
    def get_place_details(self, place_id: str):
        """Mock place details."""
        place = next((p for p in self.places_data if p["id"] == place_id), None)
        if place:
            return {"result": place, "status": "OK"}
        return {"status": "NOT_FOUND"}


class MockRoutesAPI:
    """Mock Routes API for testing."""
    
    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix
    
    def compute_routes(self, origin, destination, **kwargs):
        """Mock route computation."""
        # Return mock route with distance and duration
        return {
            "routes": [{
                "distanceMeters": 5000,
                "duration": "900s",
                "polyline": {"encodedPolyline": "mock_polyline"}
            }],
            "status": "OK"
        }
    
    def compute_route_matrix(self, origins, destinations, **kwargs):
        """Mock route matrix computation."""
        n = len(origins)
        m = len(destinations)
        # Return mock matrix
        return {
            "routeMatrix": [
                {
                    "distanceMeters": int(self.distance_matrix[i, j]) * 100,
                    "duration": f"{self.distance_matrix[i, j]}s"
                }
                for i in range(min(n, self.distance_matrix.shape[0]))
                for j in range(min(m, self.distance_matrix.shape[1]))
            ],
            "status": "OK"
        }


@pytest.fixture
def mock_places_api(sample_places):
    """Mock Places API instance."""
    return MockPlacesAPI(sample_places)


@pytest.fixture
def mock_routes_api(sample_distance_matrix):
    """Mock Routes API instance."""
    return MockRoutesAPI(sample_distance_matrix)


# ==============================================================================
# Environment Setup
# ==============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    # Set dummy API keys for tests (not real keys)
    os.environ["GOOGLE_MAPS_API_KEY"] = "TEST_API_KEY_NOT_REAL"
    yield
    # Cleanup if needed
    pass


# ==============================================================================
# Utilities
# ==============================================================================

def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype=False):
    """Assert two DataFrames are equal (helper)."""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def assert_approx_equal(a: float, b: float, tolerance: float = 0.01):
    """Assert two floats are approximately equal."""
    assert abs(a - b) < tolerance, f"{a} != {b} (tolerance={tolerance})"
