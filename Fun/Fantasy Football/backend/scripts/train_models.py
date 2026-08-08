"""Retrains the per-position projection models on all available historical data.

Usage:
    ./.venv/bin/python scripts/train_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.ml.train import train_all


def main() -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        validation_season = settings.ingestion_end_year
        print(
            f"Training on {settings.ingestion_start_year}-{settings.ingestion_end_year} "
            f"(validation season: {validation_season})..."
        )
        train_all(db, settings.ingestion_start_year, settings.ingestion_end_year, validation_season)
    finally:
        db.close()


if __name__ == "__main__":
    main()
