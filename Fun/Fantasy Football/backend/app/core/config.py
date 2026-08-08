from functools import lru_cache
from pathlib import Path

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

    ingestion_start_year: int = 2016
    ingestion_end_year: int = 2025

    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()
