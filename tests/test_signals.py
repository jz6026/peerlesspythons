from src.signals import (
    flag_coordinated_timing,
    flag_sentiment_price_divergence,
    flag_volume_bursts,
)


def test_flags_divergence_when_sentiment_and_price_disagree():
    df = pd.DataFrame(
        {
            "hour": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
            "close": [100, 100, 100, 100, 100, 95, 90, 85],  # falling
            "average_sentiment": [0, 0, 0, 0, 0, 0, 0, 0.6],  # strongly positive
            "mentions": [1] * 8,
        }
    )

    result = flag_sentiment_price_divergence(df, trend_window=6, min_sentiment=0.15, min_price_move=0.005)

    assert bool(result.loc[7, "sentiment_price_divergence"]) is True


def test_no_divergence_flag_on_near_zero_sentiment():
    df = pd.DataFrame(
        {
            "hour": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
            "close": [100, 100, 100, 100, 100, 95, 90, 85],
            "average_sentiment": [0, 0, 0, 0, 0, 0, 0, 0.05],  # too small to count
            "mentions": [1] * 8,
        }
    )

    result = flag_sentiment_price_divergence(df, trend_window=6, min_sentiment=0.15, min_price_move=0.005)

    assert bool(result.loc[7, "sentiment_price_divergence"]) is False


def test_flags_volume_burst_against_rolling_baseline():
    mentions = [2] * 24 + [50]  # sudden spike after a flat baseline
    df = pd.DataFrame(
        {
            "hour": pd.date_range("2024-01-01", periods=25, freq="h", tz="UTC"),
            "mentions": mentions,
        }
    )

    result = flag_volume_bursts(df, window=24, z_threshold=2.0)

    assert bool(result.loc[24, "volume_burst"]) is True
    assert bool(result.loc[10, "volume_burst"]) is False


def test_coordinated_timing_flags_tight_cluster_not_scattered_posts():
    reddit_df = pd.DataFrame(
        {
            "post_id": ["a", "b", "c", "d"],
            "created_at": [
                "2024-01-01 12:00:00+00:00",
                "2024-01-01 12:02:00+00:00",
                "2024-01-01 12:04:00+00:00",
                "2024-01-01 18:00:00+00:00",  # scattered, alone
            ],
        }
    )

    clusters = flag_coordinated_timing(reddit_df, window_minutes=10, min_posts=3)

    assert len(clusters) == 1
    assert clusters.loc[0, "post_count"] == 3
    assert set(clusters.loc[0, "post_ids"]) == {"a", "b", "c"}