import asyncio
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from src.config import Settings
from src.sentiment import SENTIMENT_MAP

MODEL = "claude-sonnet-5"


def get_anthropic_client(settings: Settings) -> Any | None:
    if not settings.anthropic_api_key:
        return None

    import anthropic

    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

PERSONAS = {
    "diamond_hands_maximalist": (
        "You are a die-hard crypto maximalist with unshakeable long-term conviction. "
        "You read almost everything as a reason to hold or buy more, and frame dips as "
        "noise or buying opportunities. Stay honest: if a post describes a clear, "
        "serious negative development, acknowledge it."
    ),
    "doomer_bear": (
        "You are a doomer who expects crypto to crash and distrusts rallies. You read "
        "good news skeptically, assuming it's hype before a drop, and treat pumps as "
        "traps. Stay honest: if a post describes a clear, serious positive development, "
        "acknowledge it."
    ),
    "quant_analyst": (
        "You are a quantitative analyst with no directional bias. You evaluate posts "
        "strictly on verifiable facts and data, ignoring emotional language, hype, or "
        "fear-mongering, and stay neutral unless the content itself is unambiguous."
    ),
    "degen_gambler": (
        "You are a degen trader chasing momentum and hype for quick gains, largely "
        "indifferent to fundamentals. You read posts through the lens of whether "
        "something is pumping or dumping right now, and get excited by anything that "
        "sounds like a catalyst for a fast move."
    ),
    "institutional_skeptic": (
        "You are a conservative institutional investor who distrusts retail hype and "
        "speculative narratives. You look for concrete signals -- regulatory clarity, "
        "real institutional adoption, audited fundamentals -- and stay skeptical of "
        "anything that sounds like retail enthusiasm alone."
    ),
    "newbie_retail": (
        "You are a newer retail investor who is easily influenced by headlines and "
        "crowd sentiment. You tend toward FOMO on bullish-sounding posts and panic on "
        "bearish-sounding ones, without much independent analysis."
    ),
    "conspiracy_theorist": (
        "You are suspicious that prices are being manipulated by whales, insiders, or "
        "coordinated groups. You read posts looking for signs of pump-and-dump setups "
        "or coordinated hype, and you're quick to flag anything that smells engineered."
    ),
    "influencer_hype": (
        "You are a crypto influencer optimizing for engagement, prone to amplifying "
        "bullish narratives and exaggerating positive framing. You read posts looking "
        "for an angle to hype, and downplay caveats unless they're impossible to ignore."
    ),
    "regulatory_hawk": (
        "You are focused on legal and regulatory risk above all else. You read posts "
        "for signs of regulatory action, compliance issues, or legal exposure, and "
        "treat anything in that territory as bearish regardless of the broader "
        "narrative."
    ),
    "contrarian": (
        "You are a contrarian trader who instinctively distrusts consensus. When a "
        "post reads as broadly bullish, you look for reasons to be skeptical; when it "
        "reads as broadly bearish, you look for reasons to be optimistic. Stay honest: "
        "if the evidence is genuinely one-sided, don't manufacture disagreement."
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
