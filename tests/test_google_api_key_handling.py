"""Tests for deferred Google Maps API key lookup in MCP tools."""

from __future__ import annotations

import asyncio
import importlib
import sys

import pytest

from apps.mcp_server.schemas.models import LatLng, SearchPlacesRequest
from src.routing import Location, MatrixRequest, RoutingPreference, TravelMode


def _reload_module(module_name: str):
    """Reload a module, ensuring import-time side effects are rerun."""

    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_places_module_imports_without_api_key(monkeypatch):
    """The Places helper module should import even when the key is missing."""

    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    module = _reload_module("apps.mcp_server.tools.places")

    assert module.PLACES_BASE == "https://places.googleapis.com/v1"


def test_routes_module_imports_without_api_key(monkeypatch):
    """The Routes helper module should import even when the key is missing."""

    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    module = _reload_module("apps.mcp_server.tools.routes")

    assert module.TRAFFIC_TTL_SEC == 300


def test_search_text_raises_without_api_key(monkeypatch):
    """Accessing the Places Text Search tool should fail only at call time."""

    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    places = _reload_module("apps.mcp_server.tools.places")

    request = SearchPlacesRequest(
        query="cafe",
        anchor=LatLng(lat=0.0, lng=0.0),
    )

    with pytest.raises(RuntimeError, match="Missing GOOGLE_MAPS_API_KEY"):
        asyncio.run(places.search_text(request))


def test_place_details_raises_without_api_key(monkeypatch):
    """Accessing Places details should fail only when invoked without a key."""

    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    places = _reload_module("apps.mcp_server.tools.places")

    with pytest.raises(RuntimeError, match="Missing GOOGLE_MAPS_API_KEY"):
        asyncio.run(places.place_details("sample-place-id"))


def test_compute_matrix_cached_raises_without_api_key(monkeypatch):
    """The Routes matrix helper should raise at runtime when key is missing."""

    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    routes = _reload_module("apps.mcp_server.tools.routes")

    request = MatrixRequest(
        origins=[Location(lat=0.0, lng=0.0)],
        destinations=[Location(lat=1.0, lng=1.0)],
        mode=TravelMode.DRIVE,
        routing_preference=RoutingPreference.TRAFFIC_AWARE,
    )

    with pytest.raises(RuntimeError, match="Missing GOOGLE_MAPS_API_KEY"):
        asyncio.run(routes.compute_matrix_cached(request))

