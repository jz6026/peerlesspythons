import pandas as pd

from src.aggregate import aggregate_hourly_personas
from src.config import TIMEZONE
from src.personas import PERSONAS


def compute_forward_return(market_df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df = market_df.sort_values("hour").reset_index(drop=True).copy()
    df["forward_return"] = df["close"].pct_change(periods=horizon).shift(-horizon)
    return df


def evaluate_predictiveness(
    combined_df: pd.DataFrame,
    reddit_df: pd.DataFrame,
    horizon: int = 1,
    min_mentions: int = 1,
    tz: str = TIMEZONE,
) -> pd.DataFrame:
    market_with_returns = compute_forward_return(combined_df, horizon=horizon)

    sources = {"finbert_baseline": "average_sentiment"}

    has_persona_columns = not reddit_df.empty and all(
        f"{persona}_score" in reddit_df.columns for persona in PERSONAS
    )

    if has_persona_columns:
        hourly_personas = aggregate_hourly_personas(reddit_df, PERSONAS.keys(), tz=tz)
        merged = pd.merge(
            market_with_returns,
            hourly_personas.drop(columns=["mentions"]),
            on="hour",
            how="left",
        )
        for persona in PERSONAS:
            sources[persona] = f"{persona}_avg_score"
    else:
        merged = market_with_returns

    results = []
    for source, score_col in sources.items():
        if score_col not in merged.columns:
            results.append(
                {"source": source, "correlation": None, "hit_rate": None, "sample_size": 0}
            )
            continue

        candidate = merged[merged["mentions"].fillna(0) >= min_mentions]
        valid = candidate[[score_col, "forward_return"]].dropna()

        if len(valid) < 2:
            results.append(
                {
                    "source": source,
                    "correlation": None,
                    "hit_rate": None,
                    "sample_size": len(valid),
                }
            )
            continue

        correlation = valid[score_col].corr(valid["forward_return"])

        directional = valid[valid[score_col] != 0]
        hit_rate = (
            ((directional[score_col] > 0) == (directional["forward_return"] > 0)).mean()
            if len(directional) > 0
            else None
        )

        results.append(
            {
                "source": source,
                "correlation": correlation,
                "hit_rate": hit_rate,
                "sample_size": len(valid),
            }
        )

    return pd.DataFrame(results).sort_values("correlation", ascending=False, na_position="last")
