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
    save_to_database: bool = False

    news_api_key: str | None = None

    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str | None = None

    binance_api_key: str | None = None
    binance_secret_key: str | None = None

    mysql_host: str | None = None
    mysql_user: str | None = None
    mysql_password: str | None = None
    mysql_database: str | None = None

    coins: list[str] = field(default_factory=lambda: list(COINS))


def get_settings() -> Settings:
    return Settings(
        news_api_key=os.getenv("NEWS_API_KEY"),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT"),
        binance_api_key=os.getenv("BINANCE_API_KEY"),
        binance_secret_key=os.getenv("BINANCE_SECRET_KEY"),
        mysql_host=os.getenv("MYSQL_HOST"),
        mysql_user=os.getenv("MYSQL_USER"),
        mysql_password=os.getenv("MYSQL_PASSWORD"),
        mysql_database=os.getenv("MYSQL_DATABASE"),
    )
