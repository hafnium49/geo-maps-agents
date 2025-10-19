"""FastAPI-based MCP server exposing GeoTrip tools as ChatGPT Actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas.models import (
    DetailsRequest,
    OptimizeItineraryRequest,
    OptimizeItineraryResponse,
    SearchPlacesRequest,
    SearchPlacesResponse,
)
from .tools.itinerary import score_and_sequence
from .tools.places import place_details, search_text

app = FastAPI(title="Geo Maps MCP Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assets_dir = Path(__file__).parent / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.post("/actions/search_places")
async def search_places_action(request: SearchPlacesRequest) -> Dict[str, Any]:
    places = await search_text(request)
    response = SearchPlacesResponse(places=places)

    widget_payload = {
        "widget": "geo.poiClusters",
        "props": {
            "center": request.anchor.model_dump(),
            "pois": [place.model_dump(by_alias=True) for place in response.places],
        },
        "assetsBaseUrl": "/assets",
    }

    return {
        "places": [place.model_dump() for place in response.places],
        "_meta": {"openai": {"outputTemplate": widget_payload}},
    }


@app.post("/actions/details")
async def details_action(request: DetailsRequest) -> Dict[str, Any]:
    try:
        place = await place_details(request.place_id, language=request.language)
    except Exception as exc:  # pragma: no cover - propagated to client
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    widget_payload = {
        "widget": "geo.placeCard",
        "props": place.model_dump(by_alias=True),
        "assetsBaseUrl": "/assets",
    }

    return {
        "place": place.model_dump(),
        "_meta": {"openai": {"outputTemplate": widget_payload}},
    }


@app.post("/actions/optimize_itinerary")
async def optimize_itinerary_action(request: OptimizeItineraryRequest) -> Dict[str, Any]:
    result: OptimizeItineraryResponse = await score_and_sequence(request)
    itinerary = result.itinerary

    layout_widgets = []
    layout_widgets.append(
        {
            "widget": "geo.h3Heatmap",
            "props": {
                "center": request.anchor.model_dump(),
                "hexes": [hex_bin.model_dump() for hex_bin in itinerary.hexes],
                "legend": "Localness",
            },
        }
    )
    layout_widgets.append(
        {
            "widget": "geo.poiClusters",
            "props": {
                "center": request.anchor.model_dump(),
                "pois": [poi.model_dump(by_alias=True) for poi in itinerary.scored_places],
            },
        }
    )
    layout_widgets.append(
        {
            "widget": "geo.isochroneRings",
            "props": {
                "center": request.anchor.model_dump(),
                "rings": [ring.model_dump() for ring in itinerary.rings],
            },
        }
    )
    layout_widgets.append(
        {
            "widget": "geo.routePlayback",
            "props": {
                "center": request.anchor.model_dump(),
                "path": [point.model_dump(exclude_none=True) for point in itinerary.route_path],
                "stops": [stop.model_dump(exclude_none=True) for stop in itinerary.route_stops],
            },
        }
    )

    timeline_day = itinerary.day_plans[0].date_iso if itinerary.day_plans else request.window.start_iso[:10]
    layout_widgets.append(
        {
            "widget": "geo.dayTimeline",
            "props": {
                "day": timeline_day,
                "stops": [entry.model_dump() for entry in itinerary.timeline],
            },
        }
    )

    template = {
        "layout": layout_widgets,
        "assetsBaseUrl": "/assets",
    }

    payload = result.model_dump()
    payload["_meta"] = {"openai": {"outputTemplate": template}}
    return payload


__all__ = ["app"]

