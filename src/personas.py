import asyncio
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from src.sentiment import SENTIMENT_MAP

MODEL = "claude-sonnet-5"

PERSONAS = {
    "bullish": (
        "You are a bullish crypto trader who is optimistic about the market. "
        "You tend to read ambiguous or neutral news in a positive light and "
        "highlight reasons for confidence. Stay honest: if a post is clearly "
        "bad news, say so."
    ),
    "bearish": (
        "You are a bearish crypto trader who is skeptical of the market. "
        "You tend to read ambiguous or neutral news in a negative light and "
        "highlight risks and red flags. Stay honest: if a post is clearly "
        "good news, say so."
    ),
    "neutral": (
        "You are a dispassionate financial analyst with no directional bias. "
        "You evaluate posts strictly on their factual content, ignoring hype "
        "or fear-mongering language."
    ),
}

PERSONA_INSTRUCTIONS = (
    "Read the following Reddit post about a cryptocurrency and classify its "
    "sentiment from your perspective as positive, negative, or neutral. Give "
    "a one-sentence reason.\n\nPost:\n{text}"
)


class PersonaSentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    reasoning: str


async def score_persona(
    client: Any,
    persona_key: str,
    text: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        try:
            response = await client.messages.parse(
                model=MODEL,
                max_tokens=300,
                system=PERSONAS[persona_key],
                messages=[
                    {"role": "user", "content": PERSONA_INSTRUCTIONS.format(text=text[:2000])}
                ],
                output_format=PersonaSentiment,
            )
            parsed = response.parsed_output
            return {
                "sentiment": parsed.sentiment,
                "score": SENTIMENT_MAP[parsed.sentiment],
                "reasoning": parsed.reasoning,
            }
        except Exception:
            return {"sentiment": None, "score": None, "reasoning": None}


async def add_persona_scores(df: pd.DataFrame, client: Any, concurrency: int = 5) -> pd.DataFrame:
    df = df.copy()
    semaphore = asyncio.Semaphore(concurrency)

    texts = (df["title"].fillna("") + " " + df["body"].fillna("")).tolist()

    results_by_persona = {}
    for persona_key in PERSONAS:
        results_by_persona[persona_key] = await asyncio.gather(
            *(score_persona(client, persona_key, text, semaphore) for text in texts)
        )

    for persona_key, results in results_by_persona.items():
        df[f"{persona_key}_sentiment"] = [r["sentiment"] for r in results]
        df[f"{persona_key}_score"] = [r["score"] for r in results]
        df[f"{persona_key}_reasoning"] = [r["reasoning"] for r in results]

    score_columns = [f"{persona_key}_score" for persona_key in PERSONAS]
    df["persona_divergence"] = df[score_columns].apply(
        lambda row: row.max() - row.min() if row.notna().all() else None, axis=1
    )

    return df
