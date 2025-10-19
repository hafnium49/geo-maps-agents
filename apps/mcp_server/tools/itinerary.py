"""Itinerary optimization tool built on the core GeoTrip modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.routing import (
    MatrixRequest,
    RoutingPreference,
    TravelMode,
    VRPTWConfig,
    solve_vrptw_with_fallback,
    Location as RouteLocation,
)
from src.scoring import DEFAULT_WEIGHTS, PlaceScorer, WeightConfig, get_variant_by_name

from ..schemas.models import (
    ClusterSummary,
    HexBin,
    IsochroneRing,
    ItineraryDay,
    ItineraryOutput,
    ItineraryStop,
    OptimizeItineraryRequest,
    OptimizeItineraryResponse,
    Place,
    PoiClusterPoint,
    RoutePoint,
    RouteStop,
    TimelineEntry,
)
from .places import SearchPlacesRequest, fetch_details_for_places, search_text
from .routes import compute_matrix_cached
from .spatial import cluster_hexes, dataframe_from_places, hex_dataframe_from_places


@dataclass
class _ScoredPlace:
    place: Place
    score: float
    eta_sec: int
    cluster_id: Optional[int]
    cluster_label: Optional[str]
    diversity_gain: float


def _clone_weights(config: WeightConfig) -> WeightConfig:
    return WeightConfig(
        w_rating=config.w_rating,
        w_diversity=config.w_diversity,
        w_eta=config.w_eta,
        w_open=config.w_open,
        w_crowd=config.w_crowd,
        variant_name=config.variant_name,
    )


def _weights_from_overrides(overrides) -> WeightConfig:
    base = _clone_weights(DEFAULT_WEIGHTS)
    if overrides and overrides.variant_name:
        base = _clone_weights(get_variant_by_name(overrides.variant_name))
    if overrides:
        base = WeightConfig(
            w_rating=overrides.w_rating if overrides.w_rating is not None else base.w_rating,
            w_diversity=overrides.w_diversity if overrides.w_diversity is not None else base.w_diversity,
            w_eta=overrides.w_eta if overrides.w_eta is not None else base.w_eta,
            w_open=overrides.w_open if overrides.w_open is not None else base.w_open,
            w_crowd=overrides.w_crowd if overrides.w_crowd is not None else base.w_crowd,
            variant_name=overrides.variant_name or base.variant_name,
        )
    return base


def _duration_to_seconds(value: object) -> int:
    if isinstance(value, str) and value.endswith("s"):
        try:
            return int(float(value[:-1]))
        except ValueError:
            return 0
    if isinstance(value, dict):
        seconds = value.get("seconds")
        if seconds is not None:
            return int(seconds)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _parse_matrix(destinations: pd.DataFrame, matrix: object) -> Dict[str, int]:
    elements: Iterable[Dict[str, object]] | None
    if isinstance(matrix, dict) and "routeMatrix" in matrix:
        elements = matrix.get("routeMatrix", [])  # type: ignore[assignment]
    else:
        elements = matrix  # type: ignore[assignment]

    etas: Dict[str, int] = {}
    if elements is None:
        return etas
    if not isinstance(elements, list):
        elements = list(elements)
    for element in elements:
        dest_index = element.get("destinationIndex")
        if dest_index is None:
            continue
        if dest_index >= len(destinations):
            continue
        duration = element.get("duration") or element.get("travelDuration")
        etas[destinations.iloc[int(dest_index)]["id"]] = _duration_to_seconds(duration)
    return etas


def _collect_open_periods(place: Place) -> List[Dict[str, object]]:
    periods: List[Dict[str, object]] = []
    if place.current_opening_periods:
        periods.extend(place.current_opening_periods)
    if place.secondary_opening_periods:
        periods.extend(place.secondary_opening_periods)
    if not periods and place.regular_opening_periods:
        periods.extend(place.regular_opening_periods)
    return periods


def _select_best_open_interval(place: Place, start_time: datetime, end_time: datetime) -> Optional[tuple[datetime, datetime]]:
    periods = _collect_open_periods(place)
    if not periods:
        return (start_time, end_time)

    day_anchor = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = day_anchor - timedelta(days=(day_anchor.weekday() + 1) % 7)
    candidate_bases = [week_start - timedelta(days=7), week_start, week_start + timedelta(days=7)]

    intervals: List[tuple[datetime, datetime]] = []
    seen: set[tuple[datetime, datetime]] = set()

    for base in candidate_bases:
        for period in periods:
            if not isinstance(period, dict):
                continue
            open_info = period.get("open", {}) or {}
            close_info = period.get("close", {}) or {}
            open_time = open_info.get("time")
            if open_time is None:
                continue

            try:
                open_day = int(open_info.get("day", 0))
                open_hour = int(open_time[:2])
                open_min = int(open_time[2:4])
            except (TypeError, ValueError):
                continue

            open_dt = base + timedelta(days=open_day, hours=open_hour, minutes=open_min)

            if close_info and close_info.get("time") is not None:
                close_time = close_info.get("time")
                try:
                    close_day = int(close_info.get("day", open_day))
                    close_hour = int(close_time[:2])
                    close_min = int(close_time[2:4])
                except (TypeError, ValueError):
                    close_day = open_day
                    close_hour = open_hour
                    close_min = open_min

                close_dt = base + timedelta(days=close_day, hours=close_hour, minutes=close_min)
                if close_day < open_day or (
                    close_day == open_day and (close_hour * 60 + close_min) <= (open_hour * 60 + open_min)
                ):
                    close_dt += timedelta(days=1)
            else:
                close_dt = open_dt + timedelta(hours=24)

            if close_dt <= open_dt:
                close_dt = open_dt + timedelta(hours=1)

            key = (open_dt, close_dt)
            if key in seen:
                continue
            seen.add(key)
            intervals.append(key)

    best_interval: Optional[tuple[datetime, datetime]] = None
    best_overlap = timedelta(0)

    for raw_start, raw_end in intervals:
        overlap_start = max(raw_start, start_time)
        overlap_end = min(raw_end, end_time)
        if overlap_end <= overlap_start:
            continue
        overlap_duration = overlap_end - overlap_start
        if overlap_duration > best_overlap:
            best_overlap = overlap_duration
            best_interval = (overlap_start, overlap_end)

    return best_interval


def _format_time_range(start: datetime, end: datetime) -> str:
    return f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"


def _build_isochrone_rings(anchor_lat: float, anchor_lng: float, etas: Dict[str, int], df: pd.DataFrame) -> List[IsochroneRing]:
    if not etas:
        return []

    def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        r = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lng2 - lng1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    speeds: List[float] = []
    for row in df.itertuples():
        eta = etas.get(row.id)
        if not eta or eta <= 0:
            continue
        dist = haversine(anchor_lat, anchor_lng, row.lat, row.lng)
        if dist <= 0:
            continue
        speeds.append(dist / eta)

    if not speeds:
        speeds.append(1.2)

    avg_speed = float(np.clip(np.mean(speeds), 0.8, 25.0))

    def circle(minutes: int) -> List[List[float]]:
        radius_m = avg_speed * minutes * 60
        points: List[List[float]] = []
        for i in range(32):
            theta = 2 * math.pi * i / 32
            d_lat = (radius_m / 111320.0) * math.cos(theta)
            d_lng = (radius_m / (111320.0 * math.cos(math.radians(anchor_lat)))) * math.sin(theta)
            points.append([anchor_lng + d_lng, anchor_lat + d_lat])
        return points

    rings = []
    for minutes in (15, 30, 45):
        rings.append(IsochroneRing(minutes=minutes, polygon=circle(minutes)))
    return rings


def _build_route_path(anchor_lat: float, anchor_lng: float, result_stops: List[dict], start_time: datetime) -> tuple[List[RoutePoint], List[RouteStop]]:
    points: List[RoutePoint] = [RoutePoint(lat=anchor_lat, lng=anchor_lng, t=0)]
    stops: List[RouteStop] = []

    for stop in result_stops:
        arrival = stop.get("arrival_time")
        if isinstance(arrival, datetime):
            t = int((arrival - start_time).total_seconds())
        else:
            t = 0
        points.append(RoutePoint(lat=stop.get("lat"), lng=stop.get("lng"), t=t))
        stops.append(
            RouteStop(
                lat=stop.get("lat"),
                lng=stop.get("lng"),
                name=stop.get("place_name", "Stop"),
                place_id=stop.get("place_id"),
            )
        )
    return points, stops


def _build_timeline(stops: Iterable[dict], scored_lookup: Dict[str, _ScoredPlace]) -> List[TimelineEntry]:
    entries: List[TimelineEntry] = []
    for stop in stops:
        place_id = stop.get("place_id")
        scored = scored_lookup.get(place_id)
        arrival = stop.get("arrival_time")
        depart = stop.get("departure_time")
        reason = stop.get("reason", "")
        if scored and scored.place:
            place = scored.place
            entries.append(
                TimelineEntry(
                    name=place.name,
                    arrival_iso=arrival.isoformat(timespec="seconds") if isinstance(arrival, datetime) else "",
                    depart_iso=depart.isoformat(timespec="seconds") if isinstance(depart, datetime) else "",
                    eta_sec=scored.eta_sec,
                    reason=reason,
                    maps_url=place.google_maps_uri,
                )
            )
    return entries


async def score_and_sequence(request: OptimizeItineraryRequest) -> OptimizeItineraryResponse:
    search_request = SearchPlacesRequest(
        query=request.query,
        anchor=request.anchor,
        radius_m=request.config.radius_m if hasattr(request.config, "radius_m") else 4000,
        include_types=request.config.include_types,
        language=request.language,
        max_results=request.config.max_candidates,
    )
    candidates = await search_text(search_request)
    if not candidates:
        return OptimizeItineraryResponse(
            itinerary=ItineraryOutput(
                objective="Personalized, time-window-aware plan",
                summary="No places matched the search query.",
                day_plans=[],
                clusters=[],
                scored_places=[],
                hexes=[],
                rings=[],
                route_path=[],
                route_stops=[],
                timeline=[],
            )
        )

    detailed = await fetch_details_for_places(
        candidates, language=request.language, limit=request.config.max_candidates
    )
    detailed_map = {place.id: place for place in detailed}

    seeds: List[Place] = []
    for place in candidates[: request.config.max_candidates]:
        seeds.append(detailed_map.get(place.id, place))
    seed_map = {place.id: place for place in seeds}

    places_df = dataframe_from_places(seeds)
    if places_df.empty:
        return OptimizeItineraryResponse(
            itinerary=ItineraryOutput(
                objective="Personalized, time-window-aware plan",
                summary="No candidates with sufficient data to score.",
                day_plans=[],
                clusters=[],
                scored_places=[],
                hexes=[],
                rings=[],
                route_path=[],
                route_stops=[],
                timeline=[],
            )
        )

    hex_df = hex_dataframe_from_places(places_df, request.config.h3_res)
    hex_df.attrs["h3_res"] = request.config.h3_res
    hex_df_clustered, clusters = cluster_hexes(hex_df, min_cluster_size=request.config.cluster_min_size)
    hex_df_clustered.attrs["h3_res"] = request.config.h3_res

    mode_enum = TravelMode[request.config.mode.upper()] if request.config.mode.upper() in TravelMode.__members__ else TravelMode.WALK
    pref_enum = (
        RoutingPreference[request.config.routing_preference.upper()]
        if request.config.routing_preference.upper() in RoutingPreference.__members__
        else RoutingPreference.TRAFFIC_AWARE
    )

    matrix_request = MatrixRequest(
        origins=[RouteLocation(lat=request.anchor.lat, lng=request.anchor.lng)],
        destinations=[RouteLocation(lat=float(row.lat), lng=float(row.lng)) for row in places_df.itertuples()],
        mode=mode_enum,
        routing_preference=pref_enum,
        language=request.language,
    )

    matrix = await compute_matrix_cached(matrix_request)
    etas = _parse_matrix(places_df, matrix)

    weight_config = _weights_from_overrides(request.weights)
    scorer = PlaceScorer(weights=weight_config, enable_telemetry=False)
    scored_df = scorer.score_places(places_df, etas, hex_df_clustered)

    cluster_lookup = {cluster.cluster_id: cluster.label for cluster in clusters}

    scored_places: List[_ScoredPlace] = []
    for row in scored_df.itertuples():
        original = seed_map.get(row.id)
        if original is None:
            continue
        cluster_id = int(getattr(row, "cluster_id", -1)) if hasattr(row, "cluster_id") else -1
        cluster_label = cluster_lookup.get(cluster_id)
        scored_places.append(
            _ScoredPlace(
                place=original,
                score=float(getattr(row, "score", 0.0)),
                eta_sec=int(etas.get(row.id, getattr(row, "eta", 0) or 0)),
                cluster_id=None if cluster_id == -1 else cluster_id,
                cluster_label=cluster_label,
                diversity_gain=float(getattr(row, "diversity_gain", 0.0)),
            )
        )

    scored_places.sort(key=lambda s: s.score, reverse=True)

    start_time = datetime.fromisoformat(request.window.start_iso)
    end_time = datetime.fromisoformat(request.window.end_iso)

    window_map: Dict[str, tuple[datetime, datetime]] = {}
    filtered: List[_ScoredPlace] = []
    for scored in scored_places[: min(30, len(scored_places))]:
        interval = _select_best_open_interval(scored.place, start_time, end_time)
        if interval is None:
            continue
        window_map[scored.place.id] = interval
        filtered.append(scored)

    if not filtered:
        return OptimizeItineraryResponse(
            itinerary=ItineraryOutput(
                objective="Personalized, time-window-aware plan",
                summary="No places open during the requested window.",
                day_plans=[],
                clusters=clusters,
                scored_places=[],
                hexes=[],
                rings=[],
                route_path=[],
                route_stops=[],
                timeline=[],
            )
        )

    candidates_df = pd.DataFrame(
        [
            {
                "id": s.place.id,
                "name": s.place.name,
                "lat": s.place.lat,
                "lng": s.place.lng,
                "score": s.score,
                "eta": s.eta_sec,
                "open_now": s.place.is_open_now,
                "diversity_gain": s.diversity_gain,
                "cluster_label": s.cluster_label,
                "open_time": window_map[s.place.id][0],
                "close_time": window_map[s.place.id][1],
                "maps_url": s.place.google_maps_uri,
            }
            for s in filtered
        ]
    )

    solver_config = VRPTWConfig(
        service_time_min=request.config.service_time_min,
        time_limit_sec=10,
        use_guided_local_search=True,
        verbose=False,
    )

    result = solve_vrptw_with_fallback(
        candidates=candidates_df,
        anchor_lat=request.anchor.lat,
        anchor_lng=request.anchor.lng,
        start_time=start_time,
        end_time=end_time,
        config=solver_config,
        distance_matrix_full=None,
        force_greedy=request.config.fast_mode,
    )

    itinerary_stops: List[ItineraryStop] = []
    scored_lookup = {s.place.id: s for s in filtered}

    for stop in result.stops:
        place = scored_lookup.get(stop["place_id"]).place if stop.get("place_id") in scored_lookup else None
        if not place:
            continue
        arrival = stop.get("arrival_time")
        depart = stop.get("departure_time")
        reason = stop.get("reason", "")
        window = window_map.get(place.id)
        if window and "window=" not in reason:
            reason = f"{reason}; window={_format_time_range(window[0], window[1])}" if reason else _format_time_range(window[0], window[1])
        itinerary_stops.append(
            ItineraryStop(
                place=place,
                arrival_iso=arrival.isoformat(timespec="seconds") if isinstance(arrival, datetime) else "",
                depart_iso=depart.isoformat(timespec="seconds") if isinstance(depart, datetime) else "",
                eta_sec=scored_lookup[place.id].eta_sec,
                reason=reason,
            )
        )

    day_plan = ItineraryDay(date_iso=start_time.date().isoformat(), stops=itinerary_stops)

    hexes = [HexBin(id=row.hex, value=float(row.localness)) for row in hex_df_clustered.itertuples()]
    rings = _build_isochrone_rings(request.anchor.lat, request.anchor.lng, etas, places_df)
    route_path, route_stops = _build_route_path(request.anchor.lat, request.anchor.lng, result.stops, start_time)
    timeline = _build_timeline(result.stops, scored_lookup)

    poi_points = [
        PoiClusterPoint(
            id=s.place.id,
            name=s.place.name,
            lat=s.place.lat,
            lng=s.place.lng,
            score=round(s.score, 3),
            etaSec=s.eta_sec,
            googleMapsUri=s.place.google_maps_uri,
            clusterLabel=s.cluster_label,
        )
        for s in scored_places[:40]
    ]

    itinerary = ItineraryOutput(
        objective="Personalized, time-window-aware plan",
        summary=f"{len(day_plan.stops)} stops scheduled; {len(scored_places)} candidates scored.",
        day_plans=[day_plan],
        clusters=clusters,
        scored_places=poi_points,
        hexes=hexes,
        rings=rings,
        route_path=route_path,
        route_stops=route_stops,
        timeline=timeline,
    )

    return OptimizeItineraryResponse(itinerary=itinerary)

