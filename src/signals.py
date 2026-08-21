import pandas as pd
def flag_sentiment_price_divergence(
    df: pd.DataFrame,
    trend_window: int = 6,
    min_sentiment: float = 0.15,
    min_price_move: float = 0.005,
) -> pd.DataFrame:
    df = df.sort_values("hour").reset_index(drop=True).copy()

    price_trend = df["close"].pct_change(periods=trend_window)

    sentiment_sign = df["average_sentiment"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    trend_sign = price_trend.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    disagrees = sentiment_sign != trend_sign
    meaningful = (df["average_sentiment"].abs() >= min_sentiment) & (price_trend.abs() >= min_price_move)

    df["price_trend"] = price_trend
    df["sentiment_price_divergence"] = disagrees & meaningful

    return df


def flag_volume_bursts(
    df: pd.DataFrame,
    window: int = 24,
    z_threshold: float = 2.0,
) -> pd.DataFrame:
    df = df.sort_values("hour").reset_index(drop=True).copy()

    rolling_mean = df["mentions"].rolling(window=window, min_periods=window).mean()
    rolling_std = df["mentions"].rolling(window=window, min_periods=window).std()
    z_score = (df["mentions"] - rolling_mean) / rolling_std

    df["mentions_zscore"] = z_score
    df["volume_burst"] = z_score >= z_threshold

    return df


def flag_coordinated_timing(
    reddit_df: pd.DataFrame,
    window_minutes: int = 10,
    min_posts: int = 3,
) -> pd.DataFrame:
    if reddit_df.empty:
        return pd.DataFrame(columns=["bin_start", "post_count", "post_ids"])

    df = reddit_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["bin_start"] = df["created_at"].dt.floor(f"{window_minutes}min")

    clusters = (
        df.groupby("bin_start")
        .agg(post_count=("post_id", "count"), post_ids=("post_id", list))
        .reset_index()
    )

    return clusters[clusters["post_count"] >= min_posts].reset_index(drop=True)