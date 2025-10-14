"""
Route matrix computation with guardrails, caching, and error handling.

This module encapsulates all Routes API matrix logic, providing:
- Element limit validation with helpful error messages
- Dual TTL caching (traffic-aware vs static routes)
- Exponential backoff with jitter for retries
- Streaming support placeholder for future gRPC
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

import httpx
from cachetools import TTLCache

from ..tools.fields import get_routes_matrix_mask


# -----------------------------
# Constants
# -----------------------------

class TravelMode(Enum):
    """Supported travel modes for Routes API."""
    WALK = "WALK"
    DRIVE = "DRIVE"
    BICYCLE = "BICYCLE"
    TWO_WHEELER = "TWO_WHEELER"
    TRANSIT = "TRANSIT"


class RoutingPreference(Enum):
    """Routing preference for Routes API."""
    TRAFFIC_UNAWARE = "TRAFFIC_UNAWARE"
    TRAFFIC_AWARE = "TRAFFIC_AWARE"
    TRAFFIC_AWARE_OPTIMAL = "TRAFFIC_AWARE_OPTIMAL"


# Matrix element limits from Routes API documentation
# https://developers.google.com/maps/documentation/routes/compute_route_matrix
MAX_ELEMENTS_GENERAL = 625  # 25x25 for most modes
MAX_ELEMENTS_TRANSIT = 100  # 10x10 for TRANSIT mode
MAX_ELEMENTS_TRAFFIC_AWARE_OPTIMAL = 100  # 10x10 for TRAFFIC_AWARE_OPTIMAL


# Cache TTL values (seconds)
TTL_TRAFFIC_AWARE = 5 * 60  # 5 minutes for traffic-dependent routes
TTL_STATIC = 60 * 60  # 60 minutes for non-traffic routes


# Retry configuration
MAX_RETRIES = 4
BACKOFF_BASE = 2
BACKOFF_MAX = 8


# -----------------------------
# Data Models
# -----------------------------

@dataclass
class Location:
    """Geographic location."""
    lat: float
    lng: float


@dataclass
class MatrixRequest:
    """Route matrix request parameters."""
    origins: List[Location]
    destinations: List[Location]
    mode: TravelMode
    routing_preference: RoutingPreference
    language: str = "en"


@dataclass
class MatrixLimits:
    """Element limits for a given configuration."""
    max_elements: int
    max_origins: int
    max_destinations: int
    mode: TravelMode
    routing_preference: RoutingPreference


# -----------------------------
# Cache Management
# -----------------------------

class MatrixCache:
    """Dual-TTL cache for route matrices."""
    
    def __init__(self):
        """Initialize separate caches for traffic-aware and static routes."""
        self._traffic_cache = TTLCache(maxsize=1024, ttl=TTL_TRAFFIC_AWARE)
        self._static_cache = TTLCache(maxsize=2048, ttl=TTL_STATIC)
    
    def _cache_key(self, request: MatrixRequest) -> Tuple:
        """
        Generate cache key from request.
        
        Locations are rounded to 5 decimal places (~1m precision) to allow
        cache hits for nearby requests.
        """
        origins_key = tuple((round(o.lat, 5), round(o.lng, 5)) for o in request.origins)
        dests_key = tuple((round(d.lat, 5), round(d.lng, 5)) for d in request.destinations)
        return (origins_key, dests_key, request.mode.value, request.routing_preference.value)
    
    def _is_traffic_aware(self, routing_pref: RoutingPreference) -> bool:
        """Check if routing preference uses real-time traffic data."""
        return routing_pref in (
            RoutingPreference.TRAFFIC_AWARE,
            RoutingPreference.TRAFFIC_AWARE_OPTIMAL
        )
    
    def get(self, request: MatrixRequest) -> List[Dict[str, Any]] | None:
        """Retrieve cached matrix if available."""
        key = self._cache_key(request)
        cache = (self._traffic_cache if self._is_traffic_aware(request.routing_preference)
                 else self._static_cache)
        return cache.get(key)
    
    def set(self, request: MatrixRequest, data: List[Dict[str, Any]]) -> None:
        """Store matrix in appropriate cache based on routing preference."""
        key = self._cache_key(request)
        cache = (self._traffic_cache if self._is_traffic_aware(request.routing_preference)
                 else self._static_cache)
        cache[key] = data
    
    def clear(self) -> None:
        """Clear all caches."""
        self._traffic_cache.clear()
        self._static_cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "traffic_cache": {
                "size": len(self._traffic_cache),
                "maxsize": self._traffic_cache.maxsize,
                "ttl": TTL_TRAFFIC_AWARE,
            },
            "static_cache": {
                "size": len(self._static_cache),
                "maxsize": self._static_cache.maxsize,
                "ttl": TTL_STATIC,
            },
        }


# Global cache instance
_matrix_cache = MatrixCache()


# -----------------------------
# Limit Validation
# -----------------------------

def get_matrix_limits(mode: TravelMode, routing_pref: RoutingPreference) -> MatrixLimits:
    """
    Get element limits for the given mode and routing preference.
    
    Args:
        mode: Travel mode
        routing_pref: Routing preference
        
    Returns:
        MatrixLimits with max elements and dimensions
    """
    # Determine max elements based on mode and preference
    if mode == TravelMode.TRANSIT:
        max_elements = MAX_ELEMENTS_TRANSIT
    elif routing_pref == RoutingPreference.TRAFFIC_AWARE_OPTIMAL:
        max_elements = MAX_ELEMENTS_TRAFFIC_AWARE_OPTIMAL
    else:
        max_elements = MAX_ELEMENTS_GENERAL
    
    # Calculate max dimension (assuming square matrix for simplicity)
    max_dim = int(max_elements ** 0.5)
    
    return MatrixLimits(
        max_elements=max_elements,
        max_origins=max_dim,
        max_destinations=max_dim,
        mode=mode,
        routing_preference=routing_pref,
    )


def validate_matrix_request(request: MatrixRequest) -> None:
    """
    Validate matrix request against API limits.
    
    Args:
        request: Matrix request to validate
        
    Raises:
        ValueError: If request exceeds limits with helpful suggestions
    """
    n_origins = len(request.origins)
    n_destinations = len(request.destinations)
    n_elements = n_origins * n_destinations
    
    limits = get_matrix_limits(request.mode, request.routing_preference)
    
    if n_elements > limits.max_elements:
        # Build helpful error message with suggestions
        msg_parts = [
            f"Route matrix request exceeds API limits:",
            f"  Requested: {n_origins} origins × {n_destinations} destinations = {n_elements} elements",
            f"  Maximum:   {limits.max_elements} elements for {request.mode.value} mode with {request.routing_preference.value}",
            "",
            "💡 Suggestions to fix this:",
        ]
        
        # Suggest alternative modes with higher limits
        if request.mode == TravelMode.TRANSIT:
            msg_parts.append(f"  1. Use WALK or DRIVE mode (limit: {MAX_ELEMENTS_GENERAL} elements)")
        elif request.routing_preference == RoutingPreference.TRAFFIC_AWARE_OPTIMAL:
            msg_parts.append(f"  2. Use TRAFFIC_AWARE preference (limit: {MAX_ELEMENTS_GENERAL} elements)")
        
        # Suggest batching
        batch_size = int(limits.max_elements ** 0.5)
        msg_parts.extend([
            f"  3. Reduce to {batch_size} origins and {batch_size} destinations",
            f"  4. Batch your requests (process in chunks of {batch_size}\u00D7{batch_size})",
            "  5. Pre-filter destinations to most relevant candidates",
        ])
        
        raise ValueError("\n".join(msg_parts))


# -----------------------------
# Retry Logic
# -----------------------------

def exponential_backoff_with_jitter(attempt: int) -> float:
    """
    Calculate backoff time with exponential growth and jitter.
    
    Formula: min(BACKOFF_BASE^attempt + random(0,1), BACKOFF_MAX)
    
    Args:
        attempt: Retry attempt number (0-indexed)
        
    Returns:
        Sleep time in seconds
    """
    base_delay = BACKOFF_BASE ** attempt
    jitter = random.random()  # Random value in [0, 1)
    return min(base_delay + jitter, BACKOFF_MAX)


# -----------------------------
# API Client
# -----------------------------

async def compute_route_matrix(
    request: MatrixRequest,
    api_key: str,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compute route matrix with guardrails, caching, and retries.
    
    Args:
        request: Matrix request parameters
        api_key: Google Maps API key
        use_cache: Whether to use cache (default: True)
        
    Returns:
        List of matrix elements with originIndex, destinationIndex, duration, etc.
        
    Raises:
        ValueError: If request exceeds API limits
        httpx.HTTPError: If API request fails after retries
        
    Example:
        >>> request = MatrixRequest(
        ...     origins=[Location(35.6895, 139.6917)],
        ...     destinations=[Location(35.6812, 139.7671)],
        ...     mode=TravelMode.TRANSIT,
        ...     routing_preference=RoutingPreference.TRAFFIC_AWARE,
        ... )
        >>> matrix = await compute_route_matrix(request, api_key="...")
    """
    # 1. Validate request against API limits
    validate_matrix_request(request)
    
    # 2. Check cache
    if use_cache:
        cached = _matrix_cache.get(request)
        if cached is not None:
            return cached
    
    # 3. Prepare API request
    url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
    headers = {
        "X-Goog-Api-Key": api_key,
        **get_routes_matrix_mask(),
    }
    
    body = {
        "origins": [
            {"waypoint": {"location": {"latLng": {"latitude": o.lat, "longitude": o.lng}}}}
            for o in request.origins
        ],
        "destinations": [
            {"waypoint": {"location": {"latLng": {"latitude": d.lat, "longitude": d.lng}}}}
            for d in request.destinations
        ],
        "travelMode": request.mode.value,
        "routingPreference": request.routing_preference.value,
        "languageCode": request.language,
    }
    
    # 4. Execute with retry logic
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=body, headers=headers)
                
                # Retry on 5xx errors
                if response.status_code >= 500:
                    last_error = httpx.HTTPStatusError(
                        f"Server error: {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    if attempt < MAX_RETRIES - 1:
                        sleep_time = exponential_backoff_with_jitter(attempt)
                        time.sleep(sleep_time)
                        continue
                    raise last_error
                
                response.raise_for_status()
                data = response.json()
                
                # 5. Cache successful response
                if use_cache:
                    _matrix_cache.set(request, data)
                
                return data
                
        except httpx.HTTPStatusError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1 and e.response.status_code >= 500:
                sleep_time = exponential_backoff_with_jitter(attempt)
                time.sleep(sleep_time)
                continue
            raise
            
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                sleep_time = exponential_backoff_with_jitter(attempt)
                time.sleep(sleep_time)
                continue
            raise
    
    # Should not reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Matrix computation failed after retries")


# -----------------------------
# Streaming Support (Future)
# -----------------------------

# TODO: gRPC streaming support for large matrices
# When implementing, consider:
# 1. Use grpc.aio for async streaming
# 2. Process elements as they arrive (don't wait for full response)
# 3. Maintain partial results in case of stream interruption
# 4. Document when to use streaming vs REST:
#    - REST: < 100 elements, simple use case
#    - gRPC: > 100 elements, need streaming, advanced features

async def compute_route_matrix_streaming(
    request: MatrixRequest,
    api_key: str,
) -> None:
    """
    PLACEHOLDER: Compute route matrix using gRPC streaming.
    
    Future implementation will use:
    - google.maps.routing.v2.RoutesClient
    - StreamComputeRouteMatrix RPC
    - Async iteration over response stream
    
    Benefits:
    - Handle larger matrices (beyond REST limits)
    - Process results incrementally
    - Lower latency to first result
    - Better for long-running requests
    
    Args:
        request: Matrix request parameters
        api_key: Google Maps API key
        
    Yields:
        Individual matrix elements as they're computed
        
    Example:
        >>> async for element in compute_route_matrix_streaming(request, api_key):
        ...     print(f"Origin {element['originIndex']} -> "
        ...           f"Destination {element['destinationIndex']}: "
        ...           f"{element['duration']}")
    """
    raise NotImplementedError(
        "gRPC streaming support not yet implemented. "
        "Use compute_route_matrix() for now. "
        "See https://developers.google.com/maps/documentation/routes/compute_route_matrix_streaming"
    )


# -----------------------------
# Cache Utilities
# -----------------------------

def get_cache_stats() -> Dict[str, Any]:
    """Get current cache statistics."""
    return _matrix_cache.stats()


def clear_cache() -> None:
    """Clear all route matrix caches."""
    _matrix_cache.clear()
