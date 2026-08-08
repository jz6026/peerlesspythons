from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config import Settings


def get_engine(settings: Settings) -> Engine | None:
    if not settings.save_to_database:
        return None

    creds = (
        settings.mysql_host,
        settings.mysql_user,
        settings.mysql_password,
        settings.mysql_database,
    )
    if not all(creds):
        return None

    return create_engine(
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}"
        f"@{settings.mysql_host}/{settings.mysql_database}"
    )
