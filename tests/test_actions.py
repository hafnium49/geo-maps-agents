import os

os.environ.setdefault("GOOGLE_MAPS_API_KEY", "dummy-key")

from fastapi.testclient import TestClient
import pytest

from apps.mcp_server.main import app
from apps.mcp_server.schemas.models import LatLng, Place


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    sample_places = [
        Place(
            id="poi-1",
            name="Station Cafe",
            lat=35.681236,
            lng=139.767125,
            rating=4.5,
            user_ratings_total=500,
            types=["cafe"],
            is_open_now=True,
        ),
        Place(
            id="poi-2",
            name="Evening Bar",
            lat=35.689487,
            lng=139.691711,
            rating=4.7,
            user_ratings_total=320,
            types=["bar"],
            is_open_now=False,
        ),
    ]

    async def fake_search(request):  # type: ignore[no-untyped-def]
        return sample_places

    monkeypatch.setattr("apps.mcp_server.main.search_text", fake_search)
    return TestClient(app)


def test_hex_base_map_action_returns_widgets(client: TestClient):
    payload = {
        "anchor": LatLng(lat=35.681236, lng=139.767125).model_dump(),
        "radiusM": 2000,
        "query": "cafes",
    }

    response = client.post("/actions/hex_base_map", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "hexes" in data
    assert data["_meta"]["openai"]["outputTemplate"]["layout"][0]["widget"] == "geo.h3Heatmap"


def test_hex_refine_map_action_allows_custom_metric(client: TestClient):
    payload = {
        "anchor": LatLng(lat=35.681236, lng=139.767125).model_dump(),
        "radiusM": 2000,
        "query": "bars",
        "metric": "localness",
        "topPct": 50,
    }

    response = client.post("/actions/hex_refine_map", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "refinePlan" in data
    layout = data["_meta"]["openai"]["outputTemplate"]["layout"]
    widget_names = [entry["widget"] for entry in layout]
    assert "geo.refineDots" in widget_names
