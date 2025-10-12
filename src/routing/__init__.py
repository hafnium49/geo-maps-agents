"""Routing utilities and algorithms."""

from .matrix import (
    # Core functions
    compute_route_matrix,
    compute_route_matrix_streaming,
    
    # Configuration
    get_matrix_limits,
    validate_matrix_request,
    
    # Cache management
    get_cache_stats,
    clear_cache,
    
    # Data models
    Location,
    MatrixRequest,
    MatrixLimits,
    TravelMode,
    RoutingPreference,
    
    # Constants
    MAX_ELEMENTS_GENERAL,
    MAX_ELEMENTS_TRANSIT,
    MAX_ELEMENTS_TRAFFIC_AWARE_OPTIMAL,
    TTL_TRAFFIC_AWARE,
    TTL_STATIC,
)

__all__ = [
    # Core functions
    "compute_route_matrix",
    "compute_route_matrix_streaming",
    
    # Configuration
    "get_matrix_limits",
    "validate_matrix_request",
    
    # Cache management
    "get_cache_stats",
    "clear_cache",
    
    # Data models
    "Location",
    "MatrixRequest",
    "MatrixLimits",
    "TravelMode",
    "RoutingPreference",
    
    # Constants
    "MAX_ELEMENTS_GENERAL",
    "MAX_ELEMENTS_TRANSIT",
    "MAX_ELEMENTS_TRAFFIC_AWARE_OPTIMAL",
    "TTL_TRAFFIC_AWARE",
    "TTL_STATIC",
]
