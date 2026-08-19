from unittest.mock import Mock, patch

import pandas as pd

from src.config import Settings
from src.storage import get_s3_client, load_history, upload_dataframe


def test_get_s3_client_returns_none_when_saving_disabled():
    settings = Settings(save_to_s3=False)
    assert get_s3_client(settings) is None


def test_get_s3_client_returns_none_when_creds_missing():
    settings = Settings(save_to_s3=True, s3_bucket_name="my-bucket")
    assert get_s3_client(settings) is None


def test_get_s3_client_builds_client_when_creds_present():
    settings = Settings(
        save_to_s3=True,
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        s3_bucket_name="my-bucket",
    )

    with patch("src.storage.boto3.client") as mock_client:
        result = get_s3_client(settings)

    mock_client.assert_called_once_with(
        "s3",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        region_name="us-east-1",
    )
    assert result is mock_client.return_value


def test_upload_dataframe_puts_object_with_partitioned_key():
    s3_client = Mock()
    df = pd.DataFrame({"hour": [1], "average_sentiment": [0.5]})

    key = upload_dataframe(s3_client, "my-bucket", "bitcoin", df)

    assert key.startswith("sentiment/bitcoin/dt=")
    assert key.endswith(".csv")

    s3_client.put_object.assert_called_once()
    call_kwargs = s3_client.put_object.call_args.kwargs
    assert call_kwargs["Bucket"] == "my-bucket"
    assert call_kwargs["Key"] == key
    assert "average_sentiment" in call_kwargs["Body"]


def test_load_history_returns_empty_dataframe_when_no_objects():
    client = Mock()
    client.list_objects_v2 = Mock(return_value={"Contents": []})

    result = load_history(client, "my-bucket", "bitcoin")

    assert result.empty


def test_load_history_concatenates_and_dedups_by_hour():
    frame1_csv = (
        "hour,close,average_sentiment,mentions\n"
        "2024-01-01 10:00:00+00:00,100,0.0,1\n"
        "2024-01-01 11:00:00+00:00,101,0.0,1\n"
        "2024-01-01 12:00:00+00:00,102,0.1,1\n"
    )
    frame2_csv = (
        "hour,close,average_sentiment,mentions\n"
        "2024-01-01 11:00:00+00:00,101,0.0,1\n"
        "2024-01-01 12:00:00+00:00,102,0.5,2\n"
        "2024-01-01 13:00:00+00:00,103,0.2,1\n"
    )
    bodies = {
        "sentiment/bitcoin/dt=2024-01-01/bitcoin_1.csv": frame1_csv,
        "sentiment/bitcoin/dt=2024-01-01/bitcoin_2.csv": frame2_csv,
    }

    client = Mock()
    client.list_objects_v2 = Mock(
        return_value={
            "Contents": [{"Key": key} for key in bodies],
            "IsTruncated": False,
        }
    )
    client.get_object = Mock(
        side_effect=lambda Bucket, Key: {
            "Body": Mock(read=Mock(return_value=bodies[Key].encode()))
        }
    )

    result = load_history(client, "my-bucket", "bitcoin")

    assert len(result) == 4
    hour_12 = result.loc[result["hour"] == pd.Timestamp("2024-01-01 12:00:00", tz="UTC")]
    assert hour_12["average_sentiment"].iloc[0] == 0.5
    assert hour_12["mentions"].iloc[0] == 2


def test_load_history_follows_pagination():
    csv_a = "hour,close,average_sentiment,mentions\n2024-01-01 10:00:00+00:00,100,0.0,1\n"
    csv_b = "hour,close,average_sentiment,mentions\n2024-01-02 10:00:00+00:00,110,0.0,1\n"
    bodies = {
        "sentiment/bitcoin/dt=2024-01-01/a.csv": csv_a,
        "sentiment/bitcoin/dt=2024-01-02/b.csv": csv_b,
    }

    client = Mock()
    client.list_objects_v2 = Mock(
        side_effect=[
            {
                "Contents": [{"Key": "sentiment/bitcoin/dt=2024-01-01/a.csv"}],
                "IsTruncated": True,
                "NextContinuationToken": "token123",
            },
            {
                "Contents": [{"Key": "sentiment/bitcoin/dt=2024-01-02/b.csv"}],
                "IsTruncated": False,
            },
        ]
    )
    client.get_object = Mock(
        side_effect=lambda Bucket, Key: {
            "Body": Mock(read=Mock(return_value=bodies[Key].encode()))
        }
    )

    result = load_history(client, "my-bucket", "bitcoin")

    assert len(result) == 2
    second_call_kwargs = client.list_objects_v2.call_args_list[1].kwargs
    assert second_call_kwargs["ContinuationToken"] == "token123"
