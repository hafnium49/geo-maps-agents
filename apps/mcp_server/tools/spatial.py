"""Spatial helpers built on top of :mod:`src.spatial`."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import pandas as pd

from src.scoring.scorer import score_hexes
from src.spatial import ClusteringConfig, cluster_with_fallback
from src.spatial.h3_utils import (
    Anchor,
    aggregate_hex_features,
    build_base_hex_grid,
    refine_to_resolution,
    select_refine_hexes,
)

from ..schemas.models import ClusterSummary, HexAggregate, LatLng, RefinePlan


def cluster_hexes(hex_df: pd.DataFrame, *, min_cluster_size: int = 12) -> Tuple[pd.DataFrame, List[ClusterSummary]]:
    """Cluster aggregated hex features using the enhanced fallback logic."""

    config = ClusteringConfig(min_cluster_size=min_cluster_size)
    hex_with_clusters, clusters, _diagnostics = cluster_with_fallback(hex_df, config)

    summaries = [
        ClusterSummary(
            cluster_id=c.cluster_id,
            label=c.label,
            centroid=LatLng(lat=c.centroid_lat, lng=c.centroid_lng),
            size=c.size,
            hex_ids=c.hex_ids,
        )
        for c in clusters
    ]
    return hex_with_clusters, summaries


def dataframe_from_places(places: Iterable) -> pd.DataFrame:
    """Convert a sequence of place models into the scoring dataframe."""

    rows = []
    for place in places:
        rows.append(
            dict(
                id=place.id,
                name=place.name,
                lat=place.lat,
                lng=place.lng,
                rating=place.rating or 0.0,
                nratings=place.user_ratings_total or 0,
                open_now=1 if place.is_open_now else 0,
                primary=place.primary_type or "",
                types="|".join(place.types or []),
                price_level=place.price_level,
                maps_url=place.google_maps_uri,
            )
        )
    return pd.DataFrame(rows)


def _score_ring(
    ring_df: pd.DataFrame,
    *,
    resolution: int,
    ring: str,
) -> pd.DataFrame:
    if ring_df.empty:
        return pd.DataFrame()

    scored = score_hexes(ring_df)
    scored["resolution"] = resolution
    scored["ring"] = ring
    return scored


def _merge_ring_features(
    ring_df: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    base = ring_df.rename(columns={"lat": "center_lat", "lng": "center_lng"})
    merged = base.merge(features, on="hex", how="left", suffixes=("", "_agg"))
    for column in ("poi_count", "poi_density", "reviews_sum", "open_coverage", "localness"):
        merged[column] = merged[column].fillna(0.0)
    if "center_lat_agg" in merged:
        merged["center_lat"] = merged["center_lat_agg"].fillna(merged["center_lat"])
        merged["center_lng"] = merged["center_lng_agg"].fillna(merged["center_lng"])
        merged = merged.drop(columns=[col for col in ["center_lat_agg", "center_lng_agg"] if col in merged])
    if "lat_agg" in merged:
        merged["lat"] = merged["lat_agg"].fillna(merged.get("lat"))
        merged.drop(columns=["lat_agg"], inplace=True)
    if "lng_agg" in merged:
        merged["lng"] = merged["lng_agg"].fillna(merged.get("lng"))
        merged.drop(columns=["lng_agg"], inplace=True)
    if "types_agg" in merged:
        merged["types"] = merged["types_agg"].fillna(merged.get("types"))
        merged.drop(columns=["types_agg"], inplace=True)
    return merged


def _ring_dataframe(
    places_df: pd.DataFrame,
    *,
    ring_df: pd.DataFrame,
    resolution: int,
    ring: str,
) -> pd.DataFrame:
    features = aggregate_hex_features(places_df, res=resolution)
    merged = _merge_ring_features(ring_df, features)
    return _score_ring(merged, resolution=resolution, ring=ring)


def build_base_hex_dataframe(
    places_df: pd.DataFrame,
    anchor: LatLng,
    radius_m: float,
    *,
    core_res: int = 9,
    belt_res: int = 8,
    core_radius_ratio: float = 0.6,
) -> pd.DataFrame:
    """Return scored base hex dataframe across core and belt rings."""

    base_grid = build_base_hex_grid(
        Anchor(lat=anchor.lat, lng=anchor.lng),
        radius_m,
        core_res=core_res,
        belt_res=belt_res,
        core_radius_ratio=core_radius_ratio,
    )

    core_df = base_grid[base_grid["ring"] == "core"].copy()
    belt_df = base_grid[base_grid["ring"] == "belt"].copy()

    frames: List[pd.DataFrame] = []
    if not core_df.empty:
        frames.append(
            _ring_dataframe(
                places_df,
                ring_df=core_df,
                resolution=core_res,
                ring="core",
            )
        )
    if not belt_df.empty:
        frames.append(
            _ring_dataframe(
                places_df,
                ring_df=belt_df,
                resolution=belt_res,
                ring="belt",
            )
        )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined


def dataframe_to_hex_models(hex_df: pd.DataFrame) -> List[HexAggregate]:
    models: List[HexAggregate] = []
    if hex_df.empty:
        return models

    for row in hex_df.itertuples():
        models.append(
            HexAggregate(
                hexId=row.hex,
                resolution=int(row.resolution),
                ring=getattr(row, "ring", None),
                center=LatLng(lat=float(row.center_lat), lng=float(row.center_lng)),
                poiCount=int(getattr(row, "poi_count", 0)),
                poiDensity=float(getattr(row, "poi_density", 0.0)),
                reviewsSum=float(getattr(row, "reviews_sum", 0.0)),
                openCoverage=float(getattr(row, "open_coverage", 0.0)),
                localness=float(getattr(row, "localness", 0.0)),
                score=float(getattr(row, "hex_score", 0.0)),
            )
        )
    return models


def build_refine_plan(
    places_df: pd.DataFrame,
    base_hex_df: pd.DataFrame,
    *,
    top_pct: float = 10.0,
    metric: str = "localness",
    child_res: int = 10,
) -> RefinePlan:
    """Return a :class:`RefinePlan` describing r=10 refinement."""

    parent_hexes: Sequence[str] = select_refine_hexes(base_hex_df, top_pct=top_pct, metric=metric)
    refined_df = refine_to_resolution(places_df, parent_hexes, child_res=child_res)
    if not refined_df.empty:
        refined_df["center_lat"] = refined_df["center_lat"].fillna(refined_df.get("lat"))
        refined_df["center_lng"] = refined_df["center_lng"].fillna(refined_df.get("lng"))
        refined_df = score_hexes(refined_df)
        refined_df["resolution"] = child_res
        refined_df["ring"] = "refined"

    return RefinePlan(
        base_hexes=dataframe_to_hex_models(base_hex_df),
        refined_hexes=dataframe_to_hex_models(refined_df) if refined_df is not None else [],
        refine_metric=metric,
        top_pct=top_pct,
    )

