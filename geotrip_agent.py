from __future__ import annotations

import os, json, math, time, random, asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import httpx
import numpy as np
import pandas as pd
import h3
import hdbscan
from cachetools import TTLCache

from pydantic import BaseModel, Field

# Import centralized FieldMasks and config loader
from src.tools.fields import (
    get_places_search_mask,
    get_places_details_mask,
    get_routes_matrix_mask,
    PLACES_TEXT_SEARCH_FIELDS,
    PLACES_DETAILS_FIELDS,
)
from src.tools.config_loader import get_config

# Import enhanced scoring module
from src.scoring import (
    PlaceScorer,
    WeightConfig as ScoringWeightConfig,
    select_ab_variant,
    normalize_rating,
    normalize_eta,
)

# Import enhanced routing module
from src.routing import (
    compute_route_matrix as compute_matrix_enhanced,
    MatrixRequest,
    TravelMode,
    RoutingPreference,
    Location as RouteLocation,
    solve_vrptw_with_fallback,
    VRPTWConfig,
    greedy_sequence,
)

# Import spatial clustering module
from src.spatial import (
    cluster_with_fallback,
    ClusteringConfig,
    ClusteringDiagnostics,
    ClusterInfo as SpatialClusterInfo,
    label_cluster,
)

# OpenAI Agents SDK
from agents import (
    Agent, Runner, handoff, function_tool, GuardrailFunctionOutput,
    InputGuardrail, ModelSettings
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# -----------------------------
# Constants & Config Defaults
# -----------------------------

PLACES_BASE = "https://places.googleapis.com/v1"
ROUTES_MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"

GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
if not GOOGLE_KEY:
    raise RuntimeError(
        "Missing GOOGLE_MAPS_API_KEY. "
        "Please copy .env.sample to .env and add your API key."
    )

DEFAULT_H3_RES = 9  # city-block scale
DEFAULT_CLUSTER_MIN_SIZE = 12

# scoring defaults (normalized [0,1])
DEFAULT_WEIGHTS = dict(w_rating=0.30, w_diversity=0.25, w_eta=0.20, w_open=0.15, w_crowd=0.10)

# Matrix limits and caching now handled by src.routing module
# Legacy constants kept for backward compatibility
MAX_ELEMENTS_GENERAL = 625
MAX_ELEMENTS_TRAFFIC_AWARE_OPTIMAL = 100
MAX_ELEMENTS_TRANSIT = 100

# -----------------------------
# Pydantic Types (I/O)
# -----------------------------

class Location(BaseModel):
    lat: float
    lng: float

class TimeWindow(BaseModel):
    start_iso: str  # ISO8601 local or UTC
    end_iso: str

class OptimizationConfig(BaseModel):
    h3_res: int = DEFAULT_H3_RES
    cluster_min_size: int = DEFAULT_CLUSTER_MIN_SIZE
    mode: str = "WALK"  # "DRIVE" | "WALK" | "BICYCLE" | "TWO_WHEELER" | "TRANSIT"
    routing_preference: str = "TRAFFIC_AWARE"  # "TRAFFIC_AWARE" | "TRAFFIC_AWARE_OPTIMAL"
    language: str = "en"
    max_candidates: int = 120
    include_types: Optional[List[str]] = None
    exclude_types: Optional[List[str]] = None
    city_profile: Optional[str] = None  # "dense" | "sparse" | None

class WeightConfig(BaseModel):
    w_rating: float = DEFAULT_WEIGHTS["w_rating"]
    w_diversity: float = DEFAULT_WEIGHTS["w_diversity"]
    w_eta: float = DEFAULT_WEIGHTS["w_eta"]
    w_open: float = DEFAULT_WEIGHTS["w_open"]
    w_crowd: float = DEFAULT_WEIGHTS["w_crowd"]
    variant_name: Optional[str] = "default"  # For A/B testing telemetry

class UserPrefs(BaseModel):
    # rough per-cluster-type affinities learned over time
    tag_affinity: Dict[str, float] = Field(default_factory=dict)

class PlaceLite(BaseModel):
    id: str
    name: str
    primary_type: Optional[str] = None
    types: List[str] = Field(default_factory=list)
    lat: float
    lng: float
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    price_level: Optional[int] = None
    is_open_now: Optional[bool] = None
    maps_url: Optional[str] = None

class ClusterInfo(BaseModel):
    cluster_id: int
    label: str
    hex_ids: List[str]
    centroid: Location

class ScoredPlace(BaseModel):
    place: PlaceLite
    eta_sec: int
    cluster_id: Optional[int]
    cluster_label: Optional[str]
    diversity_gain: float
    crowd_proxy: float
    score: float

class DayStop(BaseModel):
    place: PlaceLite
    arrival_iso: str
    depart_iso: str
    eta_sec: int
    reason: str

class ItineraryDay(BaseModel):
    date_iso: str
    stops: List[DayStop]

class ItineraryOutput(BaseModel):
    objective: str
    summary: str
    day_plans: List[ItineraryDay]
    clusters: List[ClusterInfo]
    deckgl_html: Optional[str] = None  # serialized HTML to write to file

# -----------------------------
# Utilities
# -----------------------------

def _fieldmask(header_fields: List[str]) -> Dict[str, str]:
    """Legacy helper - prefer using get_fieldmask_header from src.tools.fields"""
    return {"X-Goog-FieldMask": ",".join(header_fields)}

def _backoff_sleep(attempt: int):
    """
    Exponential backoff with jitter for retries.
    
    DEPRECATED: Use src.routing.matrix.exponential_backoff_with_jitter() instead.
    Kept for backward compatibility with existing code.
    
    Formula: min(2^attempt + random(0,1), 8)
    """
    time.sleep(min(2 ** attempt + random.random(), 8))

def _robust_norm(series: np.ndarray) -> np.ndarray:
    """
    Legacy normalization function for backward compatibility.
    
    DEPRECATED: Use src.scoring.percentile_norm() instead.
    This function now delegates to the new normalization module.
    """
    from src.scoring import percentile_norm
    return percentile_norm(series, low_percentile=5.0, high_percentile=95.0, invert=False)

# -----------------------------
# Google Maps Platform Tools
# -----------------------------

async def _http_post_json(url: str, json_body: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(url, json=json_body, headers=headers)
                if r.status_code >= 500:
                    _backoff_sleep(attempt)
                    continue
                r.raise_for_status()
                return r.json()
        except Exception:
            if attempt == 3: raise
            _backoff_sleep(attempt)
    raise RuntimeError("unreachable")

async def _http_get_json(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(url, headers=headers)
                if r.status_code >= 500:
                    _backoff_sleep(attempt)
                    continue
                r.raise_for_status()
                return r.json()
        except Exception:
            if attempt == 3: raise
            _backoff_sleep(attempt)
    raise RuntimeError("unreachable")

@function_tool
async def places_text_search(
    text_query: str,
    anchor: Location,
    radius_m: int = 4000,
    include_types: Optional[List[str]] = None,
    language: str = "en",
    max_results: int = 120,
) -> List[PlaceLite]:
    """
    Cost-aware discovery step: Text Search (New) with FieldMask.
    Uses centralized PLACES_TEXT_SEARCH_FIELDS from src.tools.fields.
    """
    headers = {"X-Goog-Api-Key": GOOGLE_KEY, **get_places_search_mask()}
    body = {
        "textQuery": text_query,
        "languageCode": language,
        "maxResultCount": min(max_results, 120),
        "locationBias": {
            "circle": {"center": {"latitude": anchor.lat, "longitude": anchor.lng}, "radius": radius_m}
        },
        "rankPreference": "POPULARITY"
    }
    if include_types:
        body["includedTypes"] = include_types

    data = await _http_post_json(f"{PLACES_BASE}/places:searchText", body, headers)
    out = []
    for p in data.get("places", []):
        loc = p.get("location", {})
        out.append(PlaceLite(
            id=p["id"],
            name=p["displayName"]["text"],
            primary_type=p.get("primaryType"),
            types=p.get("types", []),
            lat=loc.get("latitude"),
            lng=loc.get("longitude"),
            rating=p.get("rating"),
            user_ratings_total=p.get("userRatingCount"),
            price_level=p.get("priceLevel"),
            is_open_now=(p.get("currentOpeningHours", {}) or {}).get("openNow"),
            maps_url=p.get("googleMapsUri"),
        ))
    return out

@function_tool
async def place_details(place_id: str, language: str="en") -> PlaceLite:
    """
    Enrichment step: Details (New) with FieldMask.
    Uses centralized PLACES_DETAILS_FIELDS from src.tools.fields.
    Keeps costs low & respects the subset-of-reviews contract.
    """
    headers = {"X-Goog-Api-Key": GOOGLE_KEY, **get_places_details_mask()}
    url = f"{PLACES_BASE}/places/{place_id}"
    data = await _http_get_json(url, headers)
    loc = data.get("location", {})
    return PlaceLite(
        id=data["id"],
        name=data["displayName"]["text"],
        primary_type=data.get("primaryType"),
        types=data.get("types", []),
        lat=loc.get("latitude"),
        lng=loc.get("longitude"),
        rating=data.get("rating"),
        user_ratings_total=data.get("userRatingCount"),
        price_level=data.get("priceLevel"),
        is_open_now=(data.get("currentOpeningHours", {}) or {}).get("openNow"),
        maps_url=data.get("googleMapsUri"),
    )

def _matrix_limit(mode: str, routing_pref: str) -> int:
    """
    DEPRECATED: Use src.routing.get_matrix_limits() instead.
    Kept for backward compatibility.
    """
    if mode.upper() == "TRANSIT": return MAX_ELEMENTS_TRANSIT
    if routing_pref == "TRAFFIC_AWARE_OPTIMAL": return MAX_ELEMENTS_TRAFFIC_AWARE_OPTIMAL
    return MAX_ELEMENTS_GENERAL

@function_tool
async def route_matrix(
    origins: List[Location],
    destinations: List[Location],
    mode: str = "WALK",
    routing_preference: str = "TRAFFIC_AWARE",
    language: str = "en"
) -> List[Dict[str, Any]]:
    """
    Traffic-aware matrix with enhanced guardrails and dual-TTL caching.
    
    Now uses the enhanced src.routing module with:
    - Detailed element-limit validation with helpful error messages
    - Dual-TTL caching (5min for traffic-aware, 60min for static)
    - Exponential backoff with jitter for retries
    
    Returns list of matrix elements with originIndex/destinationIndex.
    """
    # Convert to new module's types
    route_origins = [RouteLocation(lat=o.lat, lng=o.lng) for o in origins]
    route_dests = [RouteLocation(lat=d.lat, lng=d.lng) for d in destinations]
    
    # Map mode string to enum
    try:
        travel_mode = TravelMode[mode.upper()]
    except KeyError:
        travel_mode = TravelMode.WALK
    
    # Map routing preference string to enum
    try:
        route_pref = RoutingPreference[routing_preference.upper()]
    except KeyError:
        route_pref = RoutingPreference.TRAFFIC_AWARE
    
    # Create request
    request = MatrixRequest(
        origins=route_origins,
        destinations=route_dests,
        mode=travel_mode,
        routing_preference=route_pref,
        language=language,
    )
    
    # Use enhanced compute function (with validation, caching, retries)
    return await compute_matrix_enhanced(request, api_key=GOOGLE_KEY)

# -----------------------------
# Spatial: H3, clustering, scoring
# -----------------------------

def _places_to_df(places: List[PlaceLite]) -> pd.DataFrame:
    rows = []
    for p in places:
        rows.append(dict(
            id=p.id, name=p.name, lat=p.lat, lng=p.lng, rating=p.rating or np.nan,
            nratings=p.user_ratings_total or 0, open_now=(1 if p.is_open_now else 0),
            primary=p.primary_type or "", types="|".join(p.types or [])
        ))
    return pd.DataFrame(rows)

def _localness_proxy_hex(df: pd.DataFrame, h3_res: int) -> pd.DataFrame:
    df = df.copy()
    df["hex"] = df.apply(lambda r: h3.geo_to_h3(r["lat"], r["lng"], h3_res), axis=1)
    # tourist anchors (you can tune this list)
    tourist_tokens = ("tourist_attraction","museum","amusement_park","shopping_mall","theme_park","art_gallery","aquarium","zoo","temple","shrine")
    df["is_tourist_anchor"] = df["types"].str.contains("|".join(tourist_tokens), case=False, regex=True)

    g = df.groupby("hex", as_index=False).agg(
        hex_mean_rating=("rating","mean"),
        hex_total_ratings=("nratings","sum"),
        tourist_anchor_density=("is_tourist_anchor","mean"),
        lat=("lat","mean"),
        lng=("lng","mean"),
        n=("id","count"),
    )
    # localness proxy: mean_rating * log1p(total_ratings) / (1 + tourist_anchor_density)
    g["localness"] = (g["hex_mean_rating"].fillna(0.0) * np.log1p(g["hex_total_ratings"])) / (1.0 + g["tourist_anchor_density"])
    return g

def _hdbscan_clusters(
    hex_df: pd.DataFrame,
    min_cluster_size: int
) -> Tuple[pd.DataFrame, List[ClusterInfo], ClusteringDiagnostics]:
    """
    Wrapper around enhanced clustering module with fallback logic.
    
    DEPRECATED: Use src.spatial.cluster_with_fallback() directly.
    Kept for backward compatibility.
    
    Returns:
        (hex_df_with_clusters, cluster_infos, diagnostics)
    """
    config = ClusteringConfig(
        min_cluster_size=min_cluster_size,
        min_clusters=2,
        max_clusters=10,
        enable_refitting=True,
    )
    
    hex_df_result, spatial_clusters, diagnostics = cluster_with_fallback(hex_df, config)
    
    # Convert SpatialClusterInfo to legacy ClusterInfo format
    clusters: List[ClusterInfo] = []
    for sc in spatial_clusters:
        clusters.append(ClusterInfo(
            cluster_id=sc.cluster_id,
            label=sc.label,
            hex_ids=sc.hex_ids,
            centroid=Location(lat=sc.centroid_lat, lng=sc.centroid_lng)
        ))
    
    # Log diagnostics
    if diagnostics.fallback_triggered:
        print(f"\n⚠️ Clustering Diagnostics:")
        print(f"  Fallback: {diagnostics.fallback_reason}")
        for suggestion in diagnostics.suggestions:
            print(f"  💡 {suggestion}")
    else:
        print(f"\n✅ Clustering Success:")
        print(f"  Points: {diagnostics.num_points}")
        print(f"  Clusters: {diagnostics.num_clusters}")
        print(f"  Noise: {diagnostics.num_noise}")
        if diagnostics.silhouette_score is not None:
            print(f"  Quality (silhouette): {diagnostics.silhouette_score:.3f}")
        if diagnostics.refit_attempts > 0:
            print(f"  Refit attempts: {diagnostics.refit_attempts}")
    
    return hex_df_result, clusters, diagnostics

def _label_cluster(sub: pd.DataFrame) -> str:
    """
    Legacy labeling function for backward compatibility.
    
    DEPRECATED: Use src.spatial.label_cluster() directly.
    """
    # Extract cluster ID if available
    if "cluster" in sub.columns and len(sub) > 0:
        cluster_id = sub["cluster"].iloc[0]
        return label_cluster(sub, cluster_id)
    
    # Fallback to old logic
    return " + ".join(
        [tok for tok, _cnt in pd.Series("|".join(sub.get("types", pd.Series([]))).split("|")).value_counts().head(2).items()]
    ) or "Mixed POIs"

def _score_places(places: List[PlaceLite],
                  etas_sec: Dict[str,int],
                  hex_df: pd.DataFrame,
                  weights: WeightConfig,
                  user_prefs: Optional[UserPrefs]=None) -> List[ScoredPlace]:
    """
    Score places using the enhanced scoring module with telemetry.
    
    Now delegates to src.scoring for:
    - Percentile-based normalization (5th/95th)
    - Proper ETA inversion (lower ETA = higher score)
    - Detailed telemetry logging
    """
    # Convert places to DataFrame
    df = _places_to_df(places)
    
    # Convert WeightConfig to ScoringWeightConfig
    scoring_weights = ScoringWeightConfig(
        w_rating=weights.w_rating,
        w_diversity=weights.w_diversity,
        w_eta=weights.w_eta,
        w_open=weights.w_open,
        w_crowd=weights.w_crowd,
        variant_name=getattr(weights, 'variant_name', 'custom')
    )
    
    # Convert user preferences to simple dict
    user_preferences = None
    if user_prefs and user_prefs.tag_affinity:
        user_preferences = user_prefs.tag_affinity
    
    # Use new scoring module
    from src.scoring import PlaceScorer
    scorer = PlaceScorer(weights=scoring_weights, enable_telemetry=True)
    scored_df = scorer.score_places(df, etas_sec, hex_df, user_preferences)
    
    # Convert back to ScoredPlace objects
    out: List[ScoredPlace] = []
    for r in scored_df.itertuples():
        out.append(ScoredPlace(
            place=PlaceLite(
                id=r.id,
                name=r.name,
                primary_type=r.primary,
                types=r.types.split("|") if r.types else [],
                lat=r.lat,
                lng=r.lng,
                rating=None if pd.isna(r.rating) else float(r.rating),
                user_ratings_total=int(r.nratings) if hasattr(r, 'nratings') else 0,
                price_level=None,
                is_open_now=bool(r.open_now),
                maps_url=None
            ),
            eta_sec=0 if pd.isna(r.eta) else int(r.eta),
            cluster_id=None if r.cluster_id == -1 else int(r.cluster_id),
            cluster_label=None,  # Will be filled by caller
            diversity_gain=float(r.diversity_gain),
            crowd_proxy=float(r.crowd_norm) if hasattr(r, 'crowd_norm') else 0.0,
            score=float(r.score),
        ))
    
    # Log telemetry summary
    telemetry = scorer.get_telemetry()
    if telemetry:
        print(f"\n📊 Scoring Telemetry: {len(telemetry)} places scored with variant '{scoring_weights.variant_name}'")
        # Show top 3 scores for debugging
        top_3 = sorted(telemetry, key=lambda t: t.breakdown.final_score, reverse=True)[:3]
        for i, t in enumerate(top_3, 1):
            print(f"  {i}. {t.place_name}: score={t.breakdown.final_score:.3f} "
                  f"(rating={t.breakdown.rating_score:.2f}, "
                  f"diversity={t.breakdown.diversity_score:.2f}, "
                  f"eta={t.breakdown.eta_score:.2f}, "
                  f"open={t.breakdown.open_score:.2f}, "
                  f"crowd=-{t.breakdown.crowd_score:.2f})")
    
    return out

# -----------------------------
# OR-Tools: single-day TSP-TW
# -----------------------------

def _sequence_single_day(
    stops: List[ScoredPlace],
    anchor: Location,
    window: TimeWindow,
    use_ortools: bool = True,
    service_time_min: int = 35
) -> ItineraryDay:
    """
    Sequence stops optimally using OR-Tools VRPTW or greedy fallback.
    
    Args:
        stops: List of scored places to visit
        anchor: Starting point
        window: Time window for the tour
        use_ortools: If True, use OR-Tools (else greedy). Set False for --fast mode.
        service_time_min: Minutes to spend at each stop
        
    Returns:
        ItineraryDay with optimized stop sequence
    """
    if len(stops) == 0:
        start = datetime.fromisoformat(window.start_iso)
        return ItineraryDay(date_iso=start.date().isoformat(), stops=[])
    
    # Convert stops to DataFrame for solver
    candidates = pd.DataFrame([
        {
            'id': s.place.id,
            'name': s.place.name,
            'lat': s.place.lat,
            'lng': s.place.lng,
            'score': s.score,
            'eta': s.eta_sec,
            'open_now': s.place.is_open_now,
            'diversity_gain': s.diversity_gain,
            'cluster_label': s.cluster_label,
        }
        for s in stops
    ])
    
    start_time = datetime.fromisoformat(window.start_iso)
    end_time = datetime.fromisoformat(window.end_iso)
    
    # Configure solver
    config = VRPTWConfig(
        service_time_min=service_time_min,
        time_limit_sec=10,
        use_guided_local_search=True,
        verbose=False
    )
    
    # Solve with OR-Tools (with automatic greedy fallback)
    result = solve_vrptw_with_fallback(
        candidates=candidates,
        anchor_lat=anchor.lat,
        anchor_lng=anchor.lng,
        start_time=start_time,
        end_time=end_time,
        config=config,
        distance_matrix_full=None,  # Could compute full matrix for better accuracy
        force_greedy=not use_ortools
    )
    
    # Log results
    if result.solution_found:
        print(f"\n🗺️ Route Sequencing: {result.sequence_method}")
        print(f"  Stops: {result.num_stops}/{result.num_candidates} candidates")
        print(f"  Travel time: {result.total_travel_time_sec // 60} min")
        print(f"  Service time: {result.total_service_time_sec // 60} min")
        print(f"  Total duration: {result.total_duration_sec // 60} min")
        print(f"  Solver time: {result.solver_time_sec:.3f}s")
        if result.fallback_reason:
            print(f"  Note: {result.fallback_reason}")
    else:
        print(f"\n⚠️ Sequencing failed: {result.fallback_reason}")
    
    # Convert result to DayStop format
    day_stops = []
    for stop_dict in result.stops:
        # Find original ScoredPlace for additional info
        original_stop = next((s for s in stops if s.place.id == stop_dict['place_id']), None)
        
        day_stops.append(DayStop(
            place=PlaceLite(
                id=stop_dict['place_id'],
                name=stop_dict['place_name'],
                primary_type=original_stop.place.primary_type if original_stop else None,
                types=original_stop.place.types if original_stop else [],
                lat=stop_dict['lat'],
                lng=stop_dict['lng'],
                rating=original_stop.place.rating if original_stop else None,
                user_ratings_total=original_stop.place.user_ratings_total if original_stop else None,
                price_level=original_stop.place.price_level if original_stop else None,
                is_open_now=stop_dict.get('open_now', None),
                maps_url=original_stop.place.maps_url if original_stop else None,
            ),
            arrival_iso=stop_dict['arrival_time'].isoformat(timespec="seconds"),
            depart_iso=stop_dict['departure_time'].isoformat(timespec="seconds"),
            eta_sec=stop_dict.get('eta', 0),
            reason=stop_dict['reason']
        ))
    
    return ItineraryDay(date_iso=start_time.date().isoformat(), stops=day_stops)

# -----------------------------
# deck.gl HTML (Google Maps JS)
# -----------------------------

def _render_deckgl_html(anchor: Location, scored: List[ScoredPlace], clusters: List[ClusterInfo]) -> str:
    """
    Returns a small self-contained HTML string that shows POIs + cluster centroids over a Google Map with deck.gl overlay.
    """
    pts = [
        {"name": s.place.name, "lat": s.place.lat, "lng": s.place.lng, "score": round(s.score,2),
         "eta": s.eta_sec, "cluster": s.cluster_label or ""}
        for s in scored
    ]
    cents = [{"lat": c.centroid.lat, "lng": c.centroid.lng, "label": c.label} for c in clusters]
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"><title>GeoTrip Plan</title>
    <script src="https://unpkg.com/deck.gl@^9.0.0/dist.min.js"></script>
    <script src="https://unpkg.com/@deck.gl/google-maps@^9.0.0/dist.min.js"></script>
    <script src="https://maps.googleapis.com/maps/api/js?key={GOOGLE_KEY}&v=beta&libraries=maps,marker"></script>
    <style>html,body,#map{{height:100%;margin:0}}</style>
  </head>
  <body>
    <div id="map"></div>
    <script>
      const map = new google.maps.Map(document.getElementById('map'), {{
        center: {{lat:{anchor.lat}, lng:{anchor.lng}}}, zoom: 13, mapId: undefined
      }});
      const overlay = new deck.GoogleMapsOverlay({{
        layers: [
          new deck.ScatterplotLayer({{
            id: 'poi',
            data: {json.dumps(pts)},
            getPosition: d => [d.lng, d.lat],
            getRadius: d => 15 + (d.score*10),
            pickable: true
          }}),
          new deck.TextLayer({{
            id: 'centroids',
            data: {json.dumps(cents)},
            getPosition: d => [d.lng, d.lat],
            getText: d => d.label,
            getSize: 14,
            getPixelOffset: [0, -12]
          }})
        ]
      }});
      overlay.setMap(map);
    </script>
  </body>
</html>"""

# -----------------------------
# Guardrails (ToS & sanity)
# -----------------------------

async def tos_guardrail(_ctx, _agent, input_text: str):
    text = (input_text or "").lower()
    # block common ToS-violating intents
    prohibited = [
        "openstreetmap", "osm ", " maplibre", "mapbox", "mix places data with non-google map"
    ]
    if any(tok in text for tok in prohibited):
        return GuardrailFunctionOutput(
            output_info={"reason":"Google Places content must be displayed on a Google Map and retain attribution."},
            tripwire_triggered=True
        )
    return GuardrailFunctionOutput(output_info={"ok":True}, tripwire_triggered=False)

InputTos = InputGuardrail(guardrail_function=tos_guardrail)

# -----------------------------
# Agents
# -----------------------------

# Data agent: discovery + enrichment + hex index + cluster
data_agent = Agent(
    name="Data Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You fetch Places cost-effectively (Text Search -> Details) with FieldMasks; you compute H3 features and HDBSCAN clusters.
""",
    tools=[places_text_search, place_details],
    input_guardrails=[InputTos],
    model_settings=ModelSettings(model_name="gpt-4.1-mini"),
)

# Spatial agent: travel-time, scoring, clustering
@function_tool
async def spatial_context_and_scoring(anchor: Location,
                                      window: TimeWindow,
                                      cfg: OptimizationConfig,
                                      weights: WeightConfig,
                                      seed_places: List[PlaceLite]) -> Dict[str, Any]:
    # H3 aggregation
    h3_res = cfg.h3_res
    df = _places_to_df(seed_places)
    hex_df = _localness_proxy_hex(df, h3_res)
    hex_df.attrs["h3_res"] = h3_res

    # route ETAs from anchor -> places
    origins = [anchor]
    destinations = [Location(lat=float(r.lat), lng=float(r.lng)) for r in df.itertuples()]
    matrix = await route_matrix(origins, destinations, mode=cfg.mode, routing_preference=cfg.routing_preference)
    # build {place_id: eta}
    etas: Dict[str, int] = {}
    for row in matrix:
        # originIndex always 0 here
        di = row.get("destinationIndex")
        dur = row.get("duration", "0s")
        sec = int(dur.replace("s","")) if isinstance(dur, str) and dur.endswith("s") else 0
        place_id = df.iloc[di]["id"]
        etas[place_id] = sec

    # clusters (on hex centroids) with enhanced fallback logic
    hex_df2, clusters, diagnostics = _hdbscan_clusters(hex_df, cfg.cluster_min_size)

    # score
    scored = _score_places(seed_places, etas, hex_df2, weights=weights)

    # attach cluster label by nearest hex centroid
    hex_map = hex_df2.set_index("hex")
    for s in scored:
        hx = h3.geo_to_h3(s.place.lat, s.place.lng, h3_res)
        if hx in hex_map.index:
            cid = int(hex_map.loc[hx, "cluster"])
            s.cluster_id = None if cid == -1 else cid
            s.cluster_label = None
            for c in clusters:
                if c.cluster_id == cid:
                    s.cluster_label = c.label
                    break

    # shortlist top N feasible in window
    top = sorted(scored, key=lambda x: x.score, reverse=True)[: min(30, len(scored))]
    day = _sequence_single_day(top, anchor, window)

    html = _render_deckgl_html(anchor, top, clusters)

    return ItineraryOutput(
        objective="Personalized, time-window-aware plan",
        summary=f"{len(day.stops)} stops scheduled; {len(clusters)} neighborhood patterns found.",
        day_plans=[day],
        clusters=clusters,
        deckgl_html=html
    ).dict()

spatial_agent = Agent(
    name="Spatial Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You combine travel-time context (Routes API matrix, traffic-aware) with H3 features to score and sequence POIs.
""",
    tools=[spatial_context_and_scoring],
    input_guardrails=[InputTos],
    model_settings=ModelSettings(model_name="gpt-4.1-mini"),
)

# UX agent: explain + produce output (structured)
class FinalOutput(BaseModel):
    html_filename: Optional[str]
    itinerary: ItineraryOutput

@function_tool
async def write_map_html(html: str, filename: str = "geotrip_map.html") -> str:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    return filename

ux_agent = Agent(
    name="UX Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You present a concise summary and write a deck.gl-on-Google-Maps HTML file. Always keep Places content on a Google Map.
""",
    tools=[write_map_html],
    input_guardrails=[InputTos],
    model_settings=ModelSettings(model_name="gpt-4.1-mini"),
    output_type=FinalOutput,
)

# Orchestrator with handoffs
triage_agent = Agent(
    name="GeoTrip Orchestrator",
    handoffs=[
        handoff(data_agent),
        handoff(spatial_agent),
        handoff(ux_agent),
    ],
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You run the pipeline:

1) Ask Data Agent (Text Search) for substrate near the anchor.
2) Ask Data Agent (Details) for the top K seed places.
3) Ask Spatial Agent to compute ETAs (Routes API), H3 features, HDBSCAN clusters, and scores.
4) Ask UX Agent to persist HTML overlay and return the final typed output.

Constraints:
- Use FieldMasks on all Google calls.
- Respect matrix element limits and traffic-awareness.
- Never mix Places content on a non-Google map. Ensure attribution by using the Google Map in the HTML.
- For rural sparsity, reduce min_cluster_size and fall back to score-only shortlisting if clustering is degenerate.
- Weight defaults: {json.dumps(DEFAULT_WEIGHTS)}; ensure all terms normalized to [0,1].

Return FinalOutput.
""",
    model_settings=ModelSettings(model_name="gpt-4.1"),
    output_type=FinalOutput,
)

# -----------------------------
# Example entrypoint
# -----------------------------

async def optimize_itinerary_example():
    anchor = Location(lat=35.6895, lng=139.6917)  # Tokyo Station-ish
    window = TimeWindow(
        start_iso=(datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)).isoformat(),
        end_iso=(datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)).isoformat(),
    )
    cfg = OptimizationConfig(
        h3_res=9, cluster_min_size=12, mode="TRANSIT", routing_preference="TRAFFIC_AWARE",
        language="en", include_types=["restaurant","cafe","museum","tourist_attraction"], max_candidates=100
    )
    weights = WeightConfig()
    query = "best local eats and culture spots"

    # Run the orchestration: the LLM will choose tools/handoffs
    prompt = (
        f"Anchor: {anchor}\nTimeWindow: {window}\nConfig: {cfg}\nWeights: {weights}\n"
        f"Query: {query}\n"
        "Execute the pipeline with the specified steps, and return FinalOutput."
    )
    result = await Runner.run(triage_agent, prompt)

    final = result.final_output_as(FinalOutput, raise_if_incorrect_type=False)
    # Persist the map file
    if final and final.itinerary and final.itinerary.deckgl_html:
        fname = await write_map_html(final.itinerary.deckgl_html)
        print(f"Wrote map HTML: {fname}")
    print(json.dumps(final.itinerary.dict(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(optimize_itinerary_example())
