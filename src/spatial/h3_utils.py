"""Utility helpers for working with multi-resolution H3 grids."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import h3
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Anchor:
    """Simple container for a geographic anchor point."""

    lat: float
    lng: float


def _coerce_places_dataframe(places: Iterable) -> pd.DataFrame:
    """Return a defensive copy of ``places`` as a :class:`~pandas.DataFrame`.

    The helper accepts either a dataframe or any iterable of mapping-like
    objects. Missing optional columns are filled with sensible defaults so the
    downstream aggregation logic can operate in a deterministic way.
    """

    if isinstance(places, pd.DataFrame):
        df = places.copy()
    else:
        df = pd.DataFrame(list(places))

    if df.empty:
        return df

    for column, default in (
        ("rating", np.nan),
        ("user_ratings_total", 0),
        ("is_open_now", np.nan),
        ("types", ""),
    ):
        if column not in df:
            df[column] = default

    if "types" in df:
        df["types"] = df["types"].apply(
            lambda value: "|".join(value) if isinstance(value, (list, tuple)) else (value or "")
        )

    return df


def _steps_for_radius(radius_m: float, resolution: int) -> int:
    """Return the number of hex steps required to cover ``radius_m`` metres."""

    edge_length = h3.edge_length(resolution, unit="m")
    if edge_length == 0:
        return 1
    return max(1, int(math.ceil(radius_m / edge_length)))


def build_base_hex_grid(
    anchor: Anchor,
    radius_m: float,
    *,
    core_res: int = 9,
    belt_res: int = 8,
    core_radius_ratio: float = 0.6,
) -> pd.DataFrame:
    """Generate a dual-resolution H3 grid around ``anchor``.

    The grid contains a high-resolution core (``core_res``) surrounded by a
    coarser belt (``belt_res``). Each row has the hex identifier, geographic
    centre, resolution and a ``ring`` label of either ``core`` or ``belt``.
    """

    core_radius = max(0.0, radius_m * core_radius_ratio)
    belt_radius = radius_m

    core_origin = h3.geo_to_h3(anchor.lat, anchor.lng, core_res)
    belt_origin = h3.geo_to_h3(anchor.lat, anchor.lng, belt_res)

    core_hexes = h3.k_ring(core_origin, _steps_for_radius(core_radius, core_res))
    belt_hexes = h3.k_ring(belt_origin, _steps_for_radius(belt_radius, belt_res))

    records = []
    for hex_id in core_hexes:
        lat, lng = h3.h3_to_geo(hex_id)
        records.append(
            {
                "hex": hex_id,
                "resolution": core_res,
                "ring": "core",
                "lat": lat,
                "lng": lng,
            }
        )

    for hex_id in belt_hexes:
        lat, lng = h3.h3_to_geo(hex_id)
        # Deduplicate any overlapping hexes (if belt_res == core_res this would
        # re-insert duplicates which we want to avoid).
        if any(record["hex"] == hex_id for record in records):
            continue
        records.append(
            {
                "hex": hex_id,
                "resolution": belt_res,
                "ring": "belt",
                "lat": lat,
                "lng": lng,
            }
        )

    df = pd.DataFrame(records)
    df.attrs["core_res"] = core_res
    df.attrs["belt_res"] = belt_res
    return df


def aggregate_hex_features(places: Iterable, *, res: int) -> pd.DataFrame:
    """Aggregate per-place metrics into hex-level features."""

    df = _coerce_places_dataframe(places)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "hex",
                "poi_count",
                "poi_density",
                "reviews_sum",
                "open_coverage",
                "localness",
                "center_lat",
                "center_lng",
                "lat",
                "lng",
                "types",
            ]
        )

    df = df.copy()
    df["hex"] = df.apply(lambda row: h3.geo_to_h3(row["lat"], row["lng"], res), axis=1)

    df["rating"] = df["rating"].fillna(0.0)
    df["user_ratings_total"] = df["user_ratings_total"].fillna(0.0)
    df["is_open_now"] = df["is_open_now"].fillna(0.0).astype(float)

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
    df["types"] = df["types"].fillna("")
    df["tourist_anchor"] = df["types"].str.contains("|".join(tourist_tokens), case=False, regex=True)

    grouped = df.groupby("hex")

    poi_counts = grouped.size().rename("poi_count")
    reviews_sum = grouped["user_ratings_total"].sum().rename("reviews_sum")
    open_coverage = grouped["is_open_now"].mean().rename("open_coverage")
    rating_mean = grouped["rating"].mean().rename("rating_mean")
    tourist_density = grouped["tourist_anchor"].mean().rename("tourist_density")
    types_concat = grouped["types"].apply(lambda values: "|".join(sorted({token for value in values for token in str(value).split("|") if token}))).rename("types")

    hex_area_km2 = h3.hex_area(resolution=res, unit="km^2")
    poi_density = (poi_counts / hex_area_km2).rename("poi_density")

    localness = (
        rating_mean.fillna(0.0)
        * np.log1p(reviews_sum.clip(lower=0.0))
        / (1.0 + tourist_density.fillna(0.0))
    ).rename("localness")

    centers = grouped[["lat", "lng"]].mean()

    result = (
        pd.concat(
            [
                poi_counts,
                poi_density,
                reviews_sum,
                open_coverage,
                localness,
                centers,
                types_concat,
            ],
            axis=1,
        )
        .reset_index()
        .rename(columns={"lat": "center_lat", "lng": "center_lng"})
    )

    result["lat"] = result["center_lat"]
    result["lng"] = result["center_lng"]

    result.attrs["h3_res"] = res
    return result


def select_refine_hexes(
    base_hex_df: pd.DataFrame,
    *,
    top_pct: float = 10.0,
    metric: str = "localness",
) -> Sequence[str]:
    """Select base hex IDs to refine based on ``metric`` ranking."""

    if base_hex_df.empty:
        return []

    metric_series = base_hex_df.get(metric)
    if metric_series is None:
        raise ValueError(f"Metric '{metric}' not present in dataframe")

    pct = max(0.0, min(100.0, float(top_pct)))
    if pct == 0.0:
        return []

    n_select = max(1, int(math.ceil(len(base_hex_df) * (pct / 100.0))))
    ordered = base_hex_df.sort_values(metric, ascending=False)
    return ordered.head(n_select)["hex"].tolist()


def refine_to_resolution(
    places: Iterable,
    parent_hex_ids: Sequence[str],
    *,
    child_res: int = 10,
) -> pd.DataFrame:
    """Refine the ``parent_hex_ids`` down to ``child_res`` hexes."""

    if not parent_hex_ids:
        return pd.DataFrame()

    df = _coerce_places_dataframe(places)
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["child_hex"] = df.apply(lambda row: h3.geo_to_h3(row["lat"], row["lng"], child_res), axis=1)

    mask = np.zeros(len(df), dtype=bool)
    for parent_hex in parent_hex_ids:
        parent_res = h3.h3_get_resolution(parent_hex)
        mask |= df["child_hex"].apply(lambda hx: h3.h3_to_parent(hx, parent_res) == parent_hex)

    subset = df.loc[mask]
    if subset.empty:
        return pd.DataFrame()

    aggregated = aggregate_hex_features(subset, res=child_res)
    aggregated.attrs["parent_hex_ids"] = list(parent_hex_ids)
    return aggregated


__all__ = [
    "Anchor",
    "aggregate_hex_features",
    "build_base_hex_grid",
    "refine_to_resolution",
    "select_refine_hexes",
]

