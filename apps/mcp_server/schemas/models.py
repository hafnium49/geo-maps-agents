"""Pydantic models for the Geo Maps ChatGPT App MCP server."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class LatLng(BaseModel):
    """Simple latitude/longitude container."""

    lat: float = Field(..., description="Latitude in decimal degrees")
    lng: float = Field(..., description="Longitude in decimal degrees")


class Place(BaseModel):
    """Simplified representation of a Google Place used across tools."""

    id: str
    name: str
    lat: float
    lng: float
    primary_type: Optional[str] = None
    types: List[str] = Field(default_factory=list)
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    price_level: Optional[int] = None
    is_open_now: Optional[bool] = None
    google_maps_uri: Optional[str] = Field(default=None, alias="googleMapsUri")
    current_opening_periods: Optional[List[Dict[str, Any]]] = None
    regular_opening_periods: Optional[List[Dict[str, Any]]] = None
    secondary_opening_periods: Optional[List[Dict[str, Any]]] = None

    model_config = {"populate_by_name": True}

    def merge(self, other: "Place") -> "Place":
        """Return a copy with fields from ``other`` when available."""

        data = self.model_dump()
        for key, value in other.model_dump(exclude_unset=True).items():
            if value is not None:
                data[key] = value
        return Place(**data)


class SearchPlacesRequest(BaseModel):
    query: str = Field(..., description="Text query for Places search")
    anchor: LatLng
    radius_m: int = Field(4000, ge=50, le=50000)
    include_types: Optional[List[str]] = Field(
        default=None, description="Optional list of included place types"
    )
    language: str = Field("en", description="IETF language code")
    max_results: int = Field(60, ge=1, le=120)


class SearchPlacesResponse(BaseModel):
    places: List[Place]


class DetailsRequest(BaseModel):
    place_id: str = Field(..., alias="placeId")
    language: str = Field("en", description="IETF language code")

    model_config = {"populate_by_name": True}


class DetailsResponse(BaseModel):
    place: Place


class TimeWindow(BaseModel):
    start_iso: str = Field(..., description="Start of itinerary window (ISO8601)")
    end_iso: str = Field(..., description="End of itinerary window (ISO8601)")

    @field_validator("start_iso", "end_iso")
    @classmethod
    def _validate_isoformat(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - validation error path
            raise ValueError("Invalid ISO8601 datetime string") from exc
        return value


class OptimizationConfig(BaseModel):
    h3_res: int = Field(9, ge=4, le=12)
    cluster_min_size: int = Field(12, ge=2)
    mode: str = Field("WALK", description="Travel mode (Routes API enum)")
    routing_preference: str = Field(
        "TRAFFIC_AWARE", description="Routes API routing preference"
    )
    language: str = Field("en")
    max_candidates: int = Field(80, ge=5, le=120)
    radius_m: int = Field(4000, ge=100, le=50000)
    include_types: Optional[List[str]] = None
    exclude_types: Optional[List[str]] = None
    city_profile: Optional[str] = None
    service_time_min: int = Field(35, ge=5, le=240)
    fast_mode: bool = False


class WeightOverrides(BaseModel):
    w_rating: Optional[float] = None
    w_diversity: Optional[float] = None
    w_eta: Optional[float] = None
    w_open: Optional[float] = None
    w_crowd: Optional[float] = None
    variant_name: Optional[str] = None


class OptimizeItineraryRequest(BaseModel):
    query: str
    anchor: LatLng
    window: TimeWindow
    config: OptimizationConfig = Field(default_factory=OptimizationConfig)
    weights: Optional[WeightOverrides] = None
    language: str = Field("en")


class PoiClusterPoint(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    score: Optional[float] = None
    eta_sec: Optional[int] = Field(default=None, alias="etaSec")
    google_maps_uri: Optional[str] = Field(default=None, alias="googleMapsUri")
    cluster_label: Optional[str] = Field(default=None, alias="clusterLabel")

    model_config = {"populate_by_name": True}


class HexBin(BaseModel):
    id: str
    value: float


class HexCell(BaseModel):
    """Hexagon metadata used for heatmap visualisations."""

    hex_id: str = Field(..., alias="hexId")
    resolution: int
    ring: Optional[str] = None
    center: LatLng

    model_config = {"populate_by_name": True}


class HexAggregate(BaseModel):
    """Aggregated per-hex metrics shared with widgets and actions."""

    hex_id: str = Field(..., alias="hexId")
    resolution: int
    ring: Optional[str] = None
    center: LatLng
    poi_count: int = Field(..., alias="poiCount")
    poi_density: float = Field(..., alias="poiDensity")
    reviews_sum: float = Field(..., alias="reviewsSum")
    open_coverage: float = Field(..., alias="openCoverage")
    localness: float
    score: Optional[float] = None

    model_config = {"populate_by_name": True}


class RefinePlan(BaseModel):
    """Structure describing refinement of coarse hexagons."""

    base_hexes: List[HexAggregate] = Field(default_factory=list, alias="baseHexes")
    refined_hexes: List[HexAggregate] = Field(default_factory=list, alias="refinedHexes")
    refine_metric: str = Field("localness", alias="refineMetric")
    top_pct: float = Field(10.0, alias="topPct")

    model_config = {"populate_by_name": True}


class HexBaseMapRequest(BaseModel):
    anchor: LatLng
    radius_m: int = Field(4000, ge=100, le=10000, alias="radiusM")
    query: str = Field("points of interest")
    city_profile: Optional[str] = Field(default=None, alias="cityProfile")

    model_config = {"populate_by_name": True}


class HexBaseMapResponse(BaseModel):
    hexes: List[HexAggregate]
    refine_plan: RefinePlan = Field(alias="refinePlan")

    model_config = {"populate_by_name": True}


class HexRefineMapRequest(BaseModel):
    anchor: LatLng
    radius_m: int = Field(4000, ge=100, le=10000, alias="radiusM")
    query: str = Field("points of interest")
    city_profile: Optional[str] = Field(default=None, alias="cityProfile")
    top_pct: float = Field(10.0, ge=0, le=100, alias="topPct")
    metric: str = Field("localness")

    model_config = {"populate_by_name": True}


class HexRefineMapResponse(BaseModel):
    refine_plan: RefinePlan = Field(alias="refinePlan")

    model_config = {"populate_by_name": True}


class IsochroneRing(BaseModel):
    minutes: int
    polygon: List[List[float]]


class RoutePoint(BaseModel):
    lat: float
    lng: float
    t: Optional[int] = None


class RouteStop(BaseModel):
    lat: float
    lng: float
    name: str
    place_id: Optional[str] = None


class TimelineEntry(BaseModel):
    name: str
    arrival_iso: str
    depart_iso: str
    eta_sec: int
    reason: str
    maps_url: Optional[str] = None


class ItineraryStop(BaseModel):
    place: Place
    arrival_iso: str
    depart_iso: str
    eta_sec: int
    reason: str


class ItineraryDay(BaseModel):
    date_iso: str
    stops: List[ItineraryStop]


class ClusterSummary(BaseModel):
    cluster_id: int
    label: str
    centroid: LatLng
    size: int
    hex_ids: List[str]


class ItineraryOutput(BaseModel):
    objective: str
    summary: str
    day_plans: List[ItineraryDay]
    clusters: List[ClusterSummary]
    scored_places: List[PoiClusterPoint]
    hexes: List[HexAggregate]
    rings: List[IsochroneRing]
    route_path: List[RoutePoint]
    route_stops: List[RouteStop]
    timeline: List[TimelineEntry]


class OptimizeItineraryResponse(BaseModel):
    itinerary: ItineraryOutput

