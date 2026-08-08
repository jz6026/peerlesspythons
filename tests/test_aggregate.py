import pandas as pd

from src.aggregate import aggregate_hourly_sentiment, merge_market_and_sentiment


def test_aggregate_hourly_sentiment_groups_by_hour():
    reddit_df = pd.DataFrame(
        {
            "post_id": ["a", "b", "c"],
            "created_at": pd.to_datetime(
                [
                    "2024-01-01 12:05:00+00:00",
                    "2024-01-01 12:45:00+00:00",
                    "2024-01-01 13:10:00+00:00",
                ]
            ),
            "sentiment_score": [1, -1, 0],
            "confidence": [0.8, 0.6, 0.5],
        }
    )

    hourly = aggregate_hourly_sentiment(reddit_df, tz="UTC")

    assert len(hourly) == 2
    first_hour = hourly.loc[hourly["hour"] == pd.Timestamp("2024-01-01 12:00:00", tz="UTC")]
    assert first_hour["mentions"].iloc[0] == 2
    assert first_hour["average_sentiment"].iloc[0] == 0.0


def test_merge_market_and_sentiment_fills_missing_hours_with_zero():
    market_df = pd.DataFrame(
        {
            "hour": pd.to_datetime(
                ["2024-01-01 12:00:00+00:00", "2024-01-01 13:00:00+00:00"]
            ),
            "close": [100.0, 101.0],
        }
    )
    hourly_sentiment = pd.DataFrame(
        {
            "hour": pd.to_datetime(["2024-01-01 12:00:00+00:00"]),
            "average_sentiment": [0.5],
            "mentions": [3],
            "average_confidence": [0.7],
        }
    )

    combined = merge_market_and_sentiment(market_df, hourly_sentiment)

    assert len(combined) == 2
    missing_row = combined.loc[combined["hour"] == pd.Timestamp("2024-01-01 13:00:00", tz="UTC")]
    assert missing_row["average_sentiment"].iloc[0] == 0.0
    assert missing_row["mentions"].iloc[0] == 0.0
