import pandas as pd

from src.config import TIMEZONE


def _add_hour_column(df: pd.DataFrame, tz: str = TIMEZONE) -> pd.DataFrame:
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(tz)
    df["hour"] = df["created_at"].dt.floor("h")
    return df


def aggregate_hourly_sentiment(reddit_df: pd.DataFrame, tz: str = TIMEZONE) -> pd.DataFrame:
    reddit_df = _add_hour_column(reddit_df, tz)
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


def aggregate_hourly_personas(
    reddit_df: pd.DataFrame, persona_keys, tz: str = TIMEZONE
) -> pd.DataFrame:
    reddit_df = _add_hour_column(reddit_df, tz)

    agg_kwargs = {
        f"{persona}_avg_score": (f"{persona}_score", "mean") for persona in persona_keys
    }
    agg_kwargs["persona_divergence_avg"] = ("persona_divergence", "mean")
    agg_kwargs["mentions"] = ("post_id", "count")

    return reddit_df.groupby("hour").agg(**agg_kwargs).reset_index()


def merge_market_and_sentiment(
    market_df: pd.DataFrame, hourly_sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    combined = pd.merge(market_df, hourly_sentiment_df, on="hour", how="left")

    for column in ("average_sentiment", "mentions", "average_confidence"):
        combined[column] = combined[column].fillna(0)

    return combined
