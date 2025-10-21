"""Google Places (New) helpers for the MCP server."""

from __future__ import annotations

import asyncio
import os
from typing import Iterable, List, Optional

import httpx

from src.tools import get_places_details_mask, get_places_search_mask

from ..schemas.models import Place, SearchPlacesRequest

PLACES_BASE = "https://places.googleapis.com/v1"


def _require_google_api_key() -> str:
    """Fetch the Google Maps API key from the environment at call time."""

    google_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not google_key:
        raise RuntimeError(
            "Missing GOOGLE_MAPS_API_KEY. Copy .env.sample to .env and set your key before running the MCP server."
        )
    return google_key


async def _http_post_json(url: str, payload: dict, headers: dict) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def _flatten_periods(hours: Optional[dict | list]) -> Optional[List[dict]]:
    if not hours:
        return None

    periods: List[dict] = []
    if isinstance(hours, list):
        for entry in hours:
            if isinstance(entry, dict):
                periods.extend(entry.get("periods", []) or [])
        return periods or None

    if isinstance(hours, dict):
        periods = hours.get("periods") or []
        return periods or None

    return None


def _base_place_from_payload(payload: dict) -> Place:
    location = payload.get("location", {})
    return Place(
        id=payload["id"],
        name=payload.get("displayName", {}).get("text", payload.get("id")),
        lat=location.get("latitude"),
        lng=location.get("longitude"),
        primary_type=payload.get("primaryType"),
        types=payload.get("types", []) or [],
        rating=payload.get("rating"),
        user_ratings_total=payload.get("userRatingCount"),
        price_level=payload.get("priceLevel"),
        is_open_now=((payload.get("currentOpeningHours") or {}).get("openNow")),
        google_maps_uri=payload.get("googleMapsUri"),
    )


async def search_text(request: SearchPlacesRequest) -> List[Place]:
    """Execute Places Text Search (New) with FieldMask enforcement."""

    google_key = _require_google_api_key()

    headers = {"X-Goog-Api-Key": google_key}
    headers.update(get_places_search_mask())

    body = {
        "textQuery": request.query,
        "languageCode": request.language,
        "maxResultCount": min(request.max_results, 120),
        "locationBias": {
            "circle": {
                "center": {
                    "latitude": request.anchor.lat,
                    "longitude": request.anchor.lng,
                },
                "radius": request.radius_m,
            }
        },
        "rankPreference": "POPULARITY",
    }

    if request.include_types:
        body["includedTypes"] = request.include_types

    data = await _http_post_json(f"{PLACES_BASE}/places:searchText", body, headers)
    places: List[Place] = []
    for payload in data.get("places", []):
        places.append(_base_place_from_payload(payload))
    return places


async def place_details(place_id: str, *, language: str = "en") -> Place:
    """Fetch detailed place data with FieldMask using Places Details (New)."""

    google_key = _require_google_api_key()

    headers = {"X-Goog-Api-Key": google_key}
    headers.update(get_places_details_mask())

    url = f"{PLACES_BASE}/places/{place_id}"
    params = {"languageCode": language}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        payload = response.json()

    place = _base_place_from_payload(payload)
    place.current_opening_periods = _flatten_periods(payload.get("currentOpeningHours"))
    place.regular_opening_periods = _flatten_periods(payload.get("regularOpeningHours"))
    place.secondary_opening_periods = _flatten_periods(payload.get("currentSecondaryOpeningHours"))
    return place


async def fetch_details_for_places(
    places: Iterable[Place], *, language: str = "en", limit: Optional[int] = None
) -> List[Place]:
    """Fetch details for a subset of places concurrently."""

    selected = list(places)
    if limit is not None:
        selected = selected[:limit]

    async def _fetch(p: Place) -> Place:
        detailed = await place_details(p.id, language=language)
        return p.merge(detailed)

    tasks = [asyncio.create_task(_fetch(p)) for p in selected]
    detailed_places = await asyncio.gather(*tasks, return_exceptions=True)

    merged: List[Place] = []
    for original, result in zip(selected, detailed_places):
        if isinstance(result, Exception):
            merged.append(original)
        else:
            merged.append(result)
    return merged

