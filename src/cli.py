import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from src.config import COIN_METADATA, get_settings
from src.pipeline import run_pipeline
from src.reddit_client import create_reddit_client
from src.storage import get_s3_client

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Reddit/market sentiment pipeline for a coin.")
    parser.add_argument("--coin", choices=sorted(COIN_METADATA), default="bitcoin")
    parser.add_argument("--subreddit", default="CryptoCurrency")
    parser.add_argument("--limit", type=int, default=100, help="Reddit posts per search term")
    parser.add_argument("--market-limit", type=int, default=100, help="Binance candles to fetch")
    parser.add_argument("--save-to-s3", action="store_true", help="Upload the combined dataframe to S3")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    settings.save_to_s3 = args.save_to_s3
    s3_client = get_s3_client(settings)

    async with create_reddit_client(settings) as reddit:
        combined = await run_pipeline(
            coin_key=args.coin,
            settings=settings,
            reddit=reddit,
            subreddit_name=args.subreddit,
            limit=args.limit,
            market_limit=args.market_limit,
            s3_client=s3_client,
        )

    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"{args.coin}_{timestamp}.csv"
    combined.to_csv(out_path, index=False)

    print(f"Rows: {len(combined)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
