from functools import lru_cache
from typing import Callable

import pandas as pd

SENTIMENT_MAP = {
    "positive": 1,
    "negative": -1,
    "neutral": 0,
}


@lru_cache(maxsize=None)
def load_sentiment_pipeline(model_name: str = "ProsusAI/finbert") -> Callable:
    from transformers import pipeline as hf_pipeline

    return hf_pipeline("sentiment-analysis", model=model_name)


def score_text(sentiment_pipeline: Callable, text: str) -> tuple[str | None, float | None]:
    try:
        result = sentiment_pipeline(text[:512])[0]
        return result["label"], result["score"]
    except Exception:
        return None, None


def add_sentiment_scores(df: pd.DataFrame, sentiment_pipeline: Callable) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")

    df[["sentiment", "confidence"]] = (
        df["text"].apply(lambda text: score_text(sentiment_pipeline, text)).apply(pd.Series)
    )

    df["sentiment_score"] = df["sentiment"].str.lower().map(SENTIMENT_MAP)

    return df
