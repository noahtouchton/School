from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"), extra="ignore")

    database_url: str = f"sqlite:///{DATA_DIR / 'fantasy.db'}"

    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    default_llm_provider: str = "anthropic"

    espn_league_id: int | None = None
    espn_year: int = 2025
    espn_s2: str | None = None
    espn_swid: str | None = None

    yahoo_client_id: str | None = None
    yahoo_client_secret: str | None = None
    yahoo_redirect_uri: str = "https://localhost:8000/yahoo/auth/callback"
    frontend_url: str = "http://localhost:3000"

    ingestion_start_year: int = 2016
    ingestion_end_year: int = 2025

    cors_origins: list[str] = ["http://localhost:3000"]

    # A key left blank in .env (the normal state for anything you haven't set up
    # yet, e.g. ESPN_LEAGUE_ID= straight out of .env.example) means "unset" and
    # falls back to the field's default. Without this, a blank line on a typed
    # field is an empty string that fails to parse and the app refuses to start.
    @field_validator("*", mode="before")
    @classmethod
    def _blank_uses_default(cls, value, info):
        if isinstance(value, str) and not value.strip():
            field = cls.model_fields.get(info.field_name)
            if field is not None and field.default is not PydanticUndefined:
                return field.default
        return value


@lru_cache
def get_settings() -> Settings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()
