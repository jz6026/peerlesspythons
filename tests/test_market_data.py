from unittest.mock import Mock, patch

from src.market_data import fetch_binance_data

_RAW_KLINE = [
    1700000000000,
    "100.0",
    "110.0",
    "90.0",
    "105.0",
    "10.0",
    1700003600000,
    "1050.0",
    "5",
    "6.0",
    "630.0",
    "0",
]


def test_fetch_binance_data_parses_klines():
    mock_response = Mock()
    mock_response.json.return_value = [_RAW_KLINE]
    mock_response.raise_for_status = Mock()

    with patch("src.market_data.requests.get", return_value=mock_response) as mock_get:
        result = fetch_binance_data("BTCUSDT", interval="1h", limit=1)

    mock_get.assert_called_once()
    assert list(result["open"]) == [100.0]
    assert list(result["close"]) == [105.0]
    assert "hour" in result.columns
    assert result["hour"].dt.tz is not None
