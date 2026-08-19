import argparse
import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.backtest import evaluate_predictiveness
from src.config import COIN_METADATA, get_settings
from src.personas import get_anthropic_client
from src.pipeline import run_pipeline
from src.reddit_client import create_reddit_client
from src.storage import get_s3_client, load_history

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Reddit/market sentiment pipeline for a coin.")
    parser.add_argument(
        "--coin",
        nargs="+",
        choices=sorted(COIN_METADATA),
        default=["bitcoin"],
        help="One or more coins to run, e.g. --coin bitcoin ethereum solana",
    )
    parser.add_argument(
        "--subreddit",
        default=None,
        help="Override: search only this subreddit instead of each coin's configured list")
    parser.add_argument("--limit", type=int, default=100, help="Reddit posts per search term")
    parser.add_argument("--market-limit", type=int, default=100, help="Binance candles to fetch")
    parser.add_argument("--save-to-s3", action="store_true", help="Upload the combined dataframe to S3")
    parser.add_argument(
        "--use-personas",
        action="store_true",
        help="Score posts with bullish/bearish/neutral LLM personas (claude-sonnet-5, incurs API cost)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Test whether persona sentiment predicts the forward price move (requires --use-personas)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forward return window in hours for --backtest",
    )
    parser.add_argument(
        "--backtest-history",
        action="store_true",
        help=(
            "Backtest the FinBERT baseline against all accumulated S3 history for each "
            "--coin, instead of running the pipeline. S3 reads only -- no Reddit/Binance/LLM "
            "calls. Requires AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/S3_BUCKET_NAME."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()

    if args.backtest_history:
        settings.save_to_s3 = True
        s3_client = get_s3_client(settings)
        if s3_client is None:
            print(
                "Cannot load history: AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/S3_BUCKET_NAME "
                "not configured."
            )
            return

        for coin in args.coin:
            history = load_history(s3_client, settings.s3_bucket_name, coin)
            if history.empty:
                print(f"[{coin}] No accumulated history found in S3 yet.")
                continue

            hours_with_activity = int((history["mentions"] > 0).sum())
            results = evaluate_predictiveness(history, pd.DataFrame(), horizon=args.horizon)
            print(
                f"\n[{coin}] Predictiveness vs next {args.horizon}h return "
                f"(accumulated S3 history: {len(history)} hours, "
                f"{hours_with_activity} with post activity):"
            )
            print(results.to_string(index=False))
        return

    settings.save_to_s3 = args.save_to_s3
    s3_client = get_s3_client(settings)
    anthropic_client = get_anthropic_client(settings) if args.use_personas else None

    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    async with create_reddit_client(settings) as reddit:
        for coin in args.coin:
            combined, reddit_data = await run_pipeline(
                coin_key=coin,
                settings=settings,
                reddit=reddit,
                subreddit_name=args.subreddit,
                limit=args.limit,
                market_limit=args.market_limit,
                s3_client=s3_client,
                use_personas=args.use_personas,
                anthropic_client=anthropic_client,
            )

            out_path = DATA_DIR / f"{coin}_{timestamp}.csv"
            combined.to_csv(out_path, index=False)

            print(f"[{coin}] Rows: {len(combined)}")
            print(f"[{coin}] Wrote {out_path}")

            if args.use_personas:
                persona_path = DATA_DIR / f"{coin}_personas_{timestamp}.csv"
                reddit_data.to_csv(persona_path, index=False)
                print(f"[{coin}] Wrote {persona_path}")

            if args.backtest:
                if not args.use_personas:
                    print(f"[{coin}] Skipping --backtest: requires --use-personas.")
                else:
                    results = evaluate_predictiveness(combined, reddit_data, horizon=args.horizon)
                    print(
                        f"\n[{coin}] Predictiveness vs next {args.horizon}h return "
                        "(small sample -- signal check, not a validated result):"
                    )
                    print(results.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
