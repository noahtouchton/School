"""CLI entrypoint: pulls historical seasons (cache-once) and refreshes everything that changes
during the season -- rosters, weekly stats, and injury reports (always-live, never cached).

Run this regularly during the season (e.g. weekly) to keep projections current. Follow with
scripts/generate_projections.py to regenerate projections from the freshly-ingested data.

Usage:
    ./.venv/bin/python scripts/run_ingestion.py
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.ingestion.injuries import ingest_injuries
from app.ingestion.nfl_data import ingest_season_range
from app.ingestion.rosters import refresh_current_rosters
from app.ingestion.season_teams import backfill_historical_season_teams, write_current_season_team


def main() -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        print(f"Refreshing current-season rosters/team assignments (always-live, never cached)...")
        roster_summary = refresh_current_rosters(db)
        print(f"  -> {roster_summary}")

        print(
            f"Ingesting historical seasons {settings.ingestion_start_year}-{settings.ingestion_end_year} "
            "(cache-once for completed seasons, always-live for the current one)..."
        )
        history_summary = ingest_season_range(db, settings.ingestion_start_year, settings.ingestion_end_year)
        for season, stats in history_summary.items():
            print(f"  {season}: {stats}")

        print("Backfilling season-level team history (for the new-team ML feature)...")
        for season in range(settings.ingestion_start_year, settings.ingestion_end_year + 1):
            n = backfill_historical_season_teams(db, season)
            print(f"  {season}: {n} rows")
        current_n = write_current_season_team(db)
        print(f"  current season: {current_n} rows")

        current_year = dt.date.today().year
        print(f"Refreshing injury reports for {current_year}...")
        print(f"  -> {ingest_injuries(db, current_year)} rows")
    finally:
        db.close()


if __name__ == "__main__":
    main()
