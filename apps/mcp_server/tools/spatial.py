"""Spatial helpers built on top of :mod:`src.spatial`."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from src.spatial import ClusteringConfig, cluster_with_fallback

from ..schemas.models import ClusterSummary, LatLng


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


def hex_dataframe_from_places(df: pd.DataFrame, h3_res: int) -> pd.DataFrame:
    """Project place dataframe to H3 hex grid with a localness proxy."""

    import h3
    import numpy as np

    g = df.copy()
    g["hex"] = g.apply(lambda row: h3.geo_to_h3(row["lat"], row["lng"], h3_res), axis=1)

    tourist_tokens = (
        "tourist_attraction",
        "museum",
        "amusement_park",
        "shopping_mall",
        "theme_park",
        "art_gallery",
        "aquarium",
        "zoo",
        "temple",
        "shrine",
    )
    g["is_tourist_anchor"] = g["types"].str.contains("|".join(tourist_tokens), case=False, regex=True)

    agg = g.groupby("hex", as_index=False).agg(
        hex_mean_rating=("rating", "mean"),
        hex_total_ratings=("nratings", "sum"),
        tourist_anchor_density=("is_tourist_anchor", "mean"),
        lat=("lat", "mean"),
        lng=("lng", "mean"),
        n=("id", "count"),
    )
    agg["localness"] = (
        agg["hex_mean_rating"].fillna(0.0)
        * np.log1p(agg["hex_total_ratings"])
        / (1.0 + agg["tourist_anchor_density"])
    )
    return agg


def dataframe_from_places(places: List) -> pd.DataFrame:
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

