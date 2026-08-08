import pandas as pd

from src.sentiment import SENTIMENT_MAP, add_sentiment_scores


def fake_pipeline(text):
    label = "positive" if "moon" in text.lower() else "negative"
    return [{"label": label, "score": 0.9}]


def test_add_sentiment_scores_maps_label_to_score():
    df = pd.DataFrame(
        {
            "title": ["To the moon", "Everything is crashing"],
            "body": ["", ""],
        }
    )

    result = add_sentiment_scores(df, fake_pipeline)

    assert list(result["sentiment"]) == ["positive", "negative"]
    assert list(result["sentiment_score"]) == [
        SENTIMENT_MAP["positive"],
        SENTIMENT_MAP["negative"],
    ]
    assert list(result["confidence"]) == [0.9, 0.9]


def test_add_sentiment_scores_handles_missing_body():
    df = pd.DataFrame({"title": ["To the moon"], "body": [None]})

    result = add_sentiment_scores(df, fake_pipeline)

    assert result.loc[0, "text"] == "To the moon "
    assert result.loc[0, "sentiment"] == "positive"
