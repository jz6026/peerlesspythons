import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_CANDIDATES = [_REPO_ROOT / ".env", _REPO_ROOT / "notebooks" / ".env"]

for _candidate in _ENV_CANDIDATES:
    if _candidate.exists():
        load_dotenv(_candidate)

TIMEZONE = "America/New_York"

COINS = [
    "bitcoin",
    "ethereum",
    "solana",
]

COIN_METADATA = {
    "bitcoin": {
        "symbol": "BTC",
        "binance_pair": "BTCUSDT",
        "search_terms": ["bitcoin", "btc"],
    },
    "ethereum": {
        "symbol": "ETH",
        "binance_pair": "ETHUSDT",
        "search_terms": ["ethereum", "eth"],
    },
    "solana": {
        "symbol": "SOL",
        "binance_pair": "SOLUSDT",
        "search_terms": ["solana", "sol"],
    },
}


@dataclass
class Settings:
    lookback_days: int = 30
    rolling_window: int = 24
    save_to_s3: bool = False

    news_api_key: str | None = None

    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str | None = None

    anthropic_api_key: str | None = None

    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"
    s3_bucket_name: str | None = None

    coins: list[str] = field(default_factory=lambda: list(COINS))


def get_settings() -> Settings:
    return Settings(
        news_api_key=os.getenv("NEWS_API_KEY"),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        s3_bucket_name=os.getenv("S3_BUCKET_NAME"),
    )
