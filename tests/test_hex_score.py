import pandas as pd

from src.scoring.scorer import score_hexes


def test_score_hexes_normalizes_columns():
    df = pd.DataFrame(
        {
            "hex": ["a", "b", "c"],
            "poi_density": [10.0, 5.0, 0.0],
            "reviews_sum": [100.0, 10.0, 0.0],
            "open_coverage": [0.5, 0.1, 0.0],
            "localness": [0.8, 0.2, 0.0],
            "center_lat": [35.0, 35.1, 35.2],
            "center_lng": [139.0, 139.1, 139.2],
        }
    )

    scored = score_hexes(df)

    assert set(["density_norm", "reviews_norm", "open_coverage_norm", "localness_norm", "hex_score"]).issubset(
        scored.columns
    )
    assert scored["hex_score"].between(0, 1).all()
    assert scored.loc[scored["poi_density"].idxmax(), "hex_score"] >= scored.loc[scored["poi_density"].idxmin(), "hex_score"]
