from src.config import Settings
from src.db import get_engine


def test_get_engine_returns_none_when_saving_disabled():
    settings = Settings(save_to_database=False)
    assert get_engine(settings) is None


def test_get_engine_returns_none_when_creds_missing():
    settings = Settings(save_to_database=True, mysql_host="localhost")
    assert get_engine(settings) is None


def test_get_engine_builds_engine_when_creds_present():
    settings = Settings(
        save_to_database=True,
        mysql_host="localhost",
        mysql_user="user",
        mysql_password="pw",
        mysql_database="db",
    )
    engine = get_engine(settings)
    assert engine is not None
    assert "mysql+pymysql" in str(engine.url)
