import math

import pandas as pd

from src.spatial.h3_utils import (
    Anchor,
    aggregate_hex_features,
    build_base_hex_grid,
    refine_to_resolution,
    select_refine_hexes,
)


def _sample_places() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "a",
                "lat": 35.681236,
                "lng": 139.767125,
                "rating": 4.6,
                "user_ratings_total": 1200,
                "is_open_now": 1,
                "types": "restaurant|bar",
            },
            {
                "id": "b",
                "lat": 35.689487,
                "lng": 139.691711,
                "rating": 4.4,
                "user_ratings_total": 800,
                "is_open_now": 0,
                "types": "cafe",
            },
            {
                "id": "c",
                "lat": 35.700000,
                "lng": 139.770000,
                "rating": 4.9,
                "user_ratings_total": 60,
                "is_open_now": 1,
                "types": "museum|tourist_attraction",
            },
        ]
    )


def test_build_base_hex_grid_core_and_belt():
    grid = build_base_hex_grid(Anchor(35.6895, 139.6917), 2500, core_res=9, belt_res=8)
    assert set(grid["ring"]) == {"core", "belt"}
    assert grid[grid["ring"] == "core"].iloc[0]["resolution"] == 9
    assert grid[grid["ring"] == "belt"].iloc[0]["resolution"] == 8


def test_selective_refinement_top_pct():
    places = _sample_places()
    features = aggregate_hex_features(places, res=9)
    features["hex"] = features["hex"].astype(str)
    features.loc[:, "localness"] = [0.9, 0.2, 0.5][: len(features)]

    selected = select_refine_hexes(features, top_pct=34, metric="localness")
    assert len(selected) == 2  # ceil(3 * 0.34)
    assert isinstance(selected[0], str)


def test_refine_to_resolution_returns_children():
    places = _sample_places()
    base_features = aggregate_hex_features(places, res=9)
    top_hexes = base_features.sort_values("localness", ascending=False)["hex"].head(1)
    refined = refine_to_resolution(places, top_hexes.tolist(), child_res=10)
    assert not refined.empty
    assert math.isclose(refined.attrs.get("h3_res"), 10)
