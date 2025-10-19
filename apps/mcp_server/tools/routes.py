"""Routes API helpers with TTL caching."""

from __future__ import annotations

"""Routes API helpers with TTL caching for Apps SDK actions."""

import hashlib
import json
import os
from typing import Iterable, List, Tuple

from cachetools import TTLCache

from src.routing import (
    MatrixRequest,
    RoutingPreference,
    TravelMode,
    compute_route_matrix,
    Location as RouteLocation,
)

GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

if not GOOGLE_KEY:
    raise RuntimeError(
        "Missing GOOGLE_MAPS_API_KEY. Copy .env.sample to .env and set your key before running the MCP server."
    )

TRAFFIC_TTL_SEC = 300
STATIC_TTL_SEC = 3600

_TRAFFIC_CACHE: TTLCache[str, List[dict]] = TTLCache(maxsize=128, ttl=TRAFFIC_TTL_SEC)
_STATIC_CACHE: TTLCache[str, List[dict]] = TTLCache(maxsize=128, ttl=STATIC_TTL_SEC)


def _cache_key(request: MatrixRequest) -> str:
    payload = {
        "mode": request.mode.name,
        "routing": request.routing_preference.name,
        "language": request.language,
        "origins": [(o.lat, o.lng) for o in request.origins],
        "destinations": [(d.lat, d.lng) for d in request.destinations],
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _pick_cache(preference: RoutingPreference) -> TTLCache[str, List[dict]]:
    if preference in (RoutingPreference.TRAFFIC_AWARE, RoutingPreference.TRAFFIC_AWARE_OPTIMAL):
        return _TRAFFIC_CACHE
    return _STATIC_CACHE


async def compute_matrix_cached(request: MatrixRequest) -> List[dict]:
    """Compute a route matrix with simple TTL caching."""

    cache = _pick_cache(request.routing_preference)
    key = _cache_key(request)
    if key in cache:
        return cache[key]

    matrix = await compute_route_matrix(request, api_key=GOOGLE_KEY)
    cache[key] = matrix
    return matrix


def to_route_location(points: Iterable[Tuple[float, float]]) -> List[RouteLocation]:
    return [RouteLocation(lat=lat, lng=lng) for lat, lng in points]


async def compute_matrix_from_latlng(
    origins: Iterable[Tuple[float, float]],
    destinations: Iterable[Tuple[float, float]],
    *,
    mode: str,
    routing_preference: str,
    language: str = "en",
) -> List[dict]:
    """Convenience wrapper that accepts basic tuples."""

    mode_enum = TravelMode[mode.upper()] if mode.upper() in TravelMode.__members__ else TravelMode.WALK
    pref_enum = (
        RoutingPreference[routing_preference.upper()]
        if routing_preference.upper() in RoutingPreference.__members__
        else RoutingPreference.TRAFFIC_AWARE
    )

    request = MatrixRequest(
        origins=to_route_location(list(origins)),
        destinations=to_route_location(list(destinations)),
        mode=mode_enum,
        routing_preference=pref_enum,
        language=language,
    )
    return await compute_matrix_cached(request)

