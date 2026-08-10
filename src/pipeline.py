from typing import Any

import asyncpraw
import pandas as pd

from src.aggregate import aggregate_hourly_sentiment, merge_market_and_sentiment
from src.config import COIN_METADATA, Settings
from src.market_data import fetch_binance_data
from src.reddit_client import fetch_reddit_posts
from src.sentiment import add_sentiment_scores, load_sentiment_pipeline
from src.storage import upload_dataframe

EMPTY_HOURLY_SENTIMENT_COLUMNS = ["hour", "average_sentiment", "mentions", "average_confidence"]


async def run_pipeline(
    coin_key: str,
    settings: Settings,
    reddit: asyncpraw.Reddit,
    subreddit_name: str = "CryptoCurrency",
    limit: int = 100,
    market_limit: int = 100,
    s3_client: Any | None = None,
) -> pd.DataFrame:
    metadata = COIN_METADATA[coin_key]

    market_data = fetch_binance_data(
        symbol=metadata["binance_pair"],
        interval="1h",
        limit=market_limit,
    )

    reddit_data = await fetch_reddit_posts(
        reddit,
        subreddit_name=subreddit_name,
        search_terms=metadata["search_terms"],
        limit=limit,
    )

    if reddit_data.empty:
        hourly_sentiment = pd.DataFrame(columns=EMPTY_HOURLY_SENTIMENT_COLUMNS)
    else:
        sentiment_pipeline = load_sentiment_pipeline()
        reddit_data = add_sentiment_scores(reddit_data, sentiment_pipeline)
        hourly_sentiment = aggregate_hourly_sentiment(reddit_data)

    combined = merge_market_and_sentiment(market_data, hourly_sentiment)

    if s3_client is not None and settings.s3_bucket_name:
        upload_dataframe(s3_client, settings.s3_bucket_name, coin_key, combined)

    return combined
