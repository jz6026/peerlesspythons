import math

import pandas as pd

from src.backtest import compute_forward_return, evaluate_predictiveness
from src.personas import PERSONAS


def test_compute_forward_return_shifts_by_horizon():
    market_df = pd.DataFrame(
        {
            "hour": pd.date_range("2024-01-01 12:00", periods=4, freq="h", tz="UTC"),
            "close": [100.0, 105.0, 103.0, 110.0],
        }
    )

    result = compute_forward_return(market_df, horizon=1)

    assert math.isclose(result.loc[0, "forward_return"], (105.0 - 100.0) / 100.0)
    assert math.isclose(result.loc[1, "forward_return"], (103.0 - 105.0) / 105.0)
    assert pd.isna(result.loc[3, "forward_return"])


def test_compute_forward_return_horizon_two():
    market_df = pd.DataFrame(
        {
            "hour": pd.date_range("2024-01-01 12:00", periods=4, freq="h", tz="UTC"),
            "close": [100.0, 105.0, 103.0, 110.0],
        }
    )

    result = compute_forward_return(market_df, horizon=2)

    assert math.isclose(result.loc[0, "forward_return"], (103.0 - 100.0) / 100.0)
    assert pd.isna(result.loc[2, "forward_return"])
    assert pd.isna(result.loc[3, "forward_return"])


def _make_combined_and_reddit():
    hours = pd.date_range("2024-01-01 12:00", periods=5, freq="h", tz="UTC")

    combined_df = pd.DataFrame(
        {
            "hour": hours,
            "close": [100.0, 105.0, 103.0, 110.0, 108.0],
            "average_sentiment": [0, 0, 0, 0, 0],
            "mentions": [1, 1, 1, 1, 0],
        }
    )

    other_personas = [
        p for p in PERSONAS if p not in ("diamond_hands_maximalist", "doomer_bear", "quant_analyst")
    ]

    reddit_data = {
        "post_id": ["p0", "p1", "p2", "p3"],
        "created_at": [h + pd.Timedelta(minutes=5) for h in hours[:4]],
        "diamond_hands_maximalist_score": [1, -1, 1, -1],
        "doomer_bear_score": [-1, 1, -1, 1],
        "quant_analyst_score": [0, 0, 0, 0],
    }
    for persona in other_personas:
        reddit_data[f"{persona}_score"] = [0, 0, 0, 0]
    reddit_data["persona_divergence"] = [2, 2, 2, 2]

    reddit_df = pd.DataFrame(reddit_data)

    return combined_df, reddit_df


def test_evaluate_predictiveness_perfect_and_inverse_personas():
    combined_df, reddit_df = _make_combined_and_reddit()

    results = evaluate_predictiveness(combined_df, reddit_df, horizon=1, tz="UTC").set_index(
        "source"
    )

    assert results.loc["diamond_hands_maximalist", "hit_rate"] == 1.0
    assert results.loc["diamond_hands_maximalist", "correlation"] > 0

    assert results.loc["doomer_bear", "hit_rate"] == 0.0
    assert results.loc["doomer_bear", "correlation"] < 0

    assert pd.isna(results.loc["quant_analyst", "hit_rate"])
    assert pd.isna(results.loc["quant_analyst", "correlation"])

    assert "finbert_baseline" in results.index


def test_evaluate_predictiveness_respects_min_mentions():
    combined_df, reddit_df = _make_combined_and_reddit()

    results = evaluate_predictiveness(
        combined_df, reddit_df, horizon=1, min_mentions=1, tz="UTC"
    ).set_index("source")

    assert results.loc["diamond_hands_maximalist", "sample_size"] == 4