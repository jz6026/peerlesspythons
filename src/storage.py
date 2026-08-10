import io
from datetime import datetime, timezone
from typing import Any

import boto3
import pandas as pd

from src.config import Settings


def get_s3_client(settings: Settings) -> Any | None:
    if not settings.save_to_s3:
        return None

    creds = (
        settings.aws_access_key_id,
        settings.aws_secret_access_key,
        settings.s3_bucket_name,
    )
    if not all(creds):
        return None

    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )


def upload_dataframe(s3_client: Any, bucket: str, coin_key: str, df: pd.DataFrame) -> str:
    now = datetime.now(timezone.utc)
    key = (
        f"sentiment/{coin_key}/dt={now:%Y-%m-%d}/{coin_key}_{now:%Y%m%dT%H%M%SZ}.csv"
    )

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

    return key
