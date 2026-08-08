import pandas as pd

from src.config import TIMEZONE


def aggregate_hourly_sentiment(reddit_df: pd.DataFrame, tz: str = TIMEZONE) -> pd.DataFrame:
    reddit_df = reddit_df.copy()

    reddit_df["created_at"] = (
        pd.to_datetime(reddit_df["created_at"], utc=True).dt.tz_convert(tz)
    )
    reddit_df["hour"] = reddit_df["created_at"].dt.floor("h")
    reddit_df["hour_display"] = reddit_df["hour"].dt.strftime("%Y-%m-%d %I:%M %p")

    hourly_sentiment = (
        reddit_df.groupby("hour")
        .agg(
            average_sentiment=("sentiment_score", "mean"),
            mentions=("post_id", "count"),
            average_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    return hourly_sentiment


def merge_market_and_sentiment(
    market_df: pd.DataFrame, hourly_sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    combined = pd.merge(market_df, hourly_sentiment_df, on="hour", how="left")

    for column in ("average_sentiment", "mentions", "average_confidence"):
        combined[column] = combined[column].fillna(0)

    return combined
