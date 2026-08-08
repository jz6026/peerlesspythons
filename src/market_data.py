import pandas as pd
import requests

from src.config import TIMEZONE

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

_NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def fetch_binance_data(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    response = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
    response.raise_for_status()

    market_data = pd.DataFrame(response.json(), columns=_KLINE_COLUMNS)

    market_data["open_time"] = (
        pd.to_datetime(market_data["open_time"], unit="ms", utc=True)
        .dt.tz_convert(TIMEZONE)
    )
    market_data["close_time"] = (
        pd.to_datetime(market_data["close_time"], unit="ms", utc=True)
        .dt.tz_convert(TIMEZONE)
    )
    market_data["hour"] = market_data["close_time"].dt.floor("h")

    market_data[_NUMERIC_COLUMNS] = market_data[_NUMERIC_COLUMNS].apply(pd.to_numeric)

    return market_data
