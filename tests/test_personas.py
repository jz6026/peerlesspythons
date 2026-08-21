from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from src.personas import PERSONAS, PersonaSentiment, add_persona_scores

_PERSONA_BY_PROMPT = {prompt: key for key, prompt in PERSONAS.items()}


def _make_client(sentiment_by_persona: dict, default: str = "neutral", raise_for_persona: str | None = None):
    async def fake_parse(**kwargs):
        persona_key = _PERSONA_BY_PROMPT[kwargs["system"]]
        if persona_key == raise_for_persona:
            raise RuntimeError("simulated API failure")
        sentiment = sentiment_by_persona.get(persona_key, default)
        return SimpleNamespace(
            parsed_output=PersonaSentiment(sentiment=sentiment, reasoning="because reasons")
        )

    client = MagicMock()
    client.messages.parse = AsyncMock(side_effect=fake_parse)
    return client


def _make_df():
    return pd.DataFrame({"title": ["To the moon"], "body": [""]})


async def test_add_persona_scores_adds_expected_columns():
    client = _make_client(
        {
            "diamond_hands_maximalist": "positive",
            "doomer_bear": "negative",
            "quant_analyst": "neutral",
        }
    )

    result = await add_persona_scores(_make_df(), client)

    for persona_key in PERSONAS:
        assert f"{persona_key}_sentiment" in result.columns
        assert f"{persona_key}_score" in result.columns
        assert f"{persona_key}_reasoning" in result.columns
    assert result.loc[0, "diamond_hands_maximalist_score"] == 1
    assert result.loc[0, "doomer_bear_score"] == -1
    assert result.loc[0, "quant_analyst_score"] == 0


async def test_persona_divergence_is_max_minus_min():
    client = _make_client(
        {
            "diamond_hands_maximalist": "positive",
            "doomer_bear": "negative",
            "quant_analyst": "neutral",
        }
    )

    result = await add_persona_scores(_make_df(), client)

    assert result.loc[0, "persona_divergence"] == 2


async def test_persona_divergence_is_zero_when_personas_agree():
    client = _make_client({}, default="positive")

    result = await add_persona_scores(_make_df(), client)

    assert result.loc[0, "persona_divergence"] == 0


async def test_failed_persona_call_does_not_crash_row():
    client = _make_client(
        {
            "diamond_hands_maximalist": "positive",
            "doomer_bear": "negative",
            "quant_analyst": "neutral",
        },
        raise_for_persona="doomer_bear",
    )

    result = await add_persona_scores(_make_df(), client)

    assert result.loc[0, "doomer_bear_score"] is None
    assert result.loc[0, "persona_divergence"] is None