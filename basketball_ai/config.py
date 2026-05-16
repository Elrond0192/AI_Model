"""Centralised application settings via pydantic-settings.

All configuration is read from environment variables (or a .env file when
python-dotenv is installed).  Import and use ``settings`` directly:

    from basketball_ai.config import settings
    print(settings.data_dir)
"""
from __future__ import annotations
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

        # Data
        data_dir: str = "data/sample"
        model_dir: str = "models_saved"
        data_source: str = "file"
        sql_cache_dir: str = ".sql_cache"
        sql_cache_ttl_seconds: int = 4 * 3600

        # API
        api_key: str = ""
        api_env: str = "development"
        allowed_origins: str = ""
        log_level: str = "INFO"
        log_format: str = "text"
        jwt_secret: str = ""
        jwt_algorithm: str = "HS256"
        jwt_expiry_minutes: int = 60

        # Azure SQL
        azure_sql_server: str = ""
        azure_sql_database: str = ""
        azure_sql_user: str = ""
        azure_sql_password: str = ""
        azure_sql_driver: str = "ODBC Driver 18 for SQL Server"
        azure_sql_connection_string: str = ""

        # Auth
        login_max_attempts: int = 5
        login_lockout_seconds: int = 15 * 60
        session_cookie_ttl_days: int = 7

except ImportError:
    # Fallback when pydantic-settings is not installed
    class Settings:  # type: ignore
        data_dir = "data/sample"
        model_dir = "models_saved"
        data_source = "file"
        api_key = ""
        api_env = "development"
        allowed_origins = ""
        log_level = "INFO"
        log_format = "text"

settings = Settings()
