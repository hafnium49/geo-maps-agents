"""FastAPI-based MCP server exposing GeoTrip tools as ChatGPT Actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .schemas.models import (
    HexBaseMapRequest,
    HexBaseMapResponse,
    HexRefineMapRequest,
    HexRefineMapResponse,
    DetailsRequest,
    OptimizeItineraryRequest,
    OptimizeItineraryResponse,
    SearchPlacesRequest,
    SearchPlacesResponse,
)
from .tools.itinerary import score_and_sequence
from .tools.places import place_details, search_text
from .tools.spatial import build_base_hex_dataframe, build_refine_plan, dataframe_from_places
from src.tools.config_loader import ConfigLoader

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

_manifest_path = Path(__file__).resolve().parents[2] / "app.manifest.json"
_manifest_data: Optional[Dict[str, Any]]
if _manifest_path.exists():
    with _manifest_path.open("r", encoding="utf-8") as manifest_file:
        _manifest_data = json.load(manifest_file)
else:
    _manifest_data = None


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/.well-known/app-manifest.json")
async def app_manifest() -> JSONResponse:
    if _manifest_data is None:
        raise HTTPException(status_code=404, detail="App manifest not found.")
    return JSONResponse(_manifest_data)


def _load_profile(name: Optional[str]) -> Dict[str, Any]:
    if name:
        return ConfigLoader.load_city_profile(name)
    return ConfigLoader.load_default_or_env_profile()


def _h3_settings(profile: Dict[str, Any]) -> Dict[str, Any]:
    h3_cfg = profile.get("h3", {})
    refine_cfg = profile.get("refine", {})
    return {
        "core_res": h3_cfg.get("base_res_core") or h3_cfg.get("primary_resolution", 9),
        "belt_res": h3_cfg.get("base_res_belt") or h3_cfg.get("fallback_resolution", 8),
        "core_ratio": refine_cfg.get("core_radius_ratio", 0.6),
        "refine_res": h3_cfg.get("refine_res") or h3_cfg.get("refine_resolution", 10),
        "top_pct": refine_cfg.get("top_pct") or profile.get("refine_top_pct", 10),
        "metric": refine_cfg.get("metric") or profile.get("refine_metric", "localness"),
    }


def _search_settings(profile: Dict[str, Any]) -> Dict[str, Any]:
    search_cfg = profile.get("search", {})
    return {
        "language": search_cfg.get("language", "en"),
        "max_results": search_cfg.get("max_candidates", 80),
    }


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


@app.post("/actions/hex_base_map")
async def hex_base_map_action(request: HexBaseMapRequest) -> Dict[str, Any]:
    profile = _load_profile(request.city_profile)
    search_cfg = _search_settings(profile)
    settings = _h3_settings(profile)

    search_request = SearchPlacesRequest(
        query=request.query,
        anchor=request.anchor,
        radius_m=request.radius_m,
        language=search_cfg["language"],
        max_results=search_cfg["max_results"],
    )
    places = await search_text(search_request)
    places_df = dataframe_from_places(places)

    base_hex_df = build_base_hex_dataframe(
        places_df,
        request.anchor,
        request.radius_m,
        core_res=settings["core_res"],
        belt_res=settings["belt_res"],
        core_radius_ratio=settings["core_ratio"],
    )
    refine_plan = build_refine_plan(
        places_df,
        base_hex_df,
        top_pct=settings["top_pct"],
        metric=settings["metric"],
        child_res=settings["refine_res"],
    )

    response = HexBaseMapResponse(hexes=refine_plan.base_hexes, refine_plan=refine_plan)

    heatmap_hexes = [hex.model_dump(by_alias=True) for hex in response.hexes]
    refined_hexes = [hex.model_dump(by_alias=True) for hex in response.refine_plan.refined_hexes]

    layout_widgets = [
        {
            "widget": "geo.h3Heatmap",
            "props": {
                "center": request.anchor.model_dump(),
                "hexes": heatmap_hexes,
                "legend": "Localness × Density",
            },
        }
    ]

    if refined_hexes:
        layout_widgets.append(
            {
                "widget": "geo.refineOutlines",
                "props": {
                    "center": request.anchor.model_dump(),
                    "hexes": refined_hexes,
                },
            }
        )
        layout_widgets.append(
            {
                "widget": "geo.refineDots",
                "props": {
                    "center": request.anchor.model_dump(),
                    "hexes": refined_hexes,
                },
            }
        )

    template = {"layout": layout_widgets, "assetsBaseUrl": "/assets"}
    payload = response.model_dump(by_alias=True)
    payload["_meta"] = {"openai": {"outputTemplate": template}}
    return payload


@app.post("/actions/hex_refine_map")
async def hex_refine_map_action(request: HexRefineMapRequest) -> Dict[str, Any]:
    profile = _load_profile(request.city_profile)
    search_cfg = _search_settings(profile)
    settings = _h3_settings(profile)

    search_request = SearchPlacesRequest(
        query=request.query,
        anchor=request.anchor,
        radius_m=request.radius_m,
        language=search_cfg["language"],
        max_results=search_cfg["max_results"],
    )
    places = await search_text(search_request)
    places_df = dataframe_from_places(places)

    base_hex_df = build_base_hex_dataframe(
        places_df,
        request.anchor,
        request.radius_m,
        core_res=settings["core_res"],
        belt_res=settings["belt_res"],
        core_radius_ratio=settings["core_ratio"],
    )
    top_pct = request.top_pct if request.top_pct is not None else settings["top_pct"]
    metric = request.metric or settings["metric"]
    refine_plan = build_refine_plan(
        places_df,
        base_hex_df,
        top_pct=top_pct,
        metric=metric,
        child_res=settings["refine_res"],
    )

    response = HexRefineMapResponse(refine_plan=refine_plan)

    heatmap_hexes = [hex.model_dump(by_alias=True) for hex in refine_plan.base_hexes]
    refined_hexes = [hex.model_dump(by_alias=True) for hex in refine_plan.refined_hexes]

    layout_widgets = [
        {
            "widget": "geo.h3Heatmap",
            "props": {
                "center": request.anchor.model_dump(),
                "hexes": heatmap_hexes,
                "legend": "Localness × Density",
            },
        }
    ]

    if refined_hexes:
        layout_widgets.append(
            {
                "widget": "geo.refineOutlines",
                "props": {
                    "center": request.anchor.model_dump(),
                    "hexes": refined_hexes,
                },
            }
        )
        layout_widgets.append(
            {
                "widget": "geo.refineDots",
                "props": {
                    "center": request.anchor.model_dump(),
                    "hexes": refined_hexes,
                },
            }
        )

    template = {"layout": layout_widgets, "assetsBaseUrl": "/assets"}
    payload = response.model_dump(by_alias=True)
    payload["_meta"] = {"openai": {"outputTemplate": template}}
    return payload


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
                "hexes": [hex_bin.model_dump(by_alias=True) for hex_bin in itinerary.hexes],
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

