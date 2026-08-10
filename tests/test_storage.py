from unittest.mock import Mock, patch

import pandas as pd

from src.config import Settings
from src.storage import get_s3_client, upload_dataframe


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
