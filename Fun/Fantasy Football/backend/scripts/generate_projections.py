"""Refreshes consensus ADP and (re)generates weekly projections for the given season/week.

Usage:
    ./.venv/bin/python scripts/generate_projections.py [season] [week]
    (defaults to the current calendar year, week 1)
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import SessionLocal
from app.ingestion.adp import ingest_consensus_adp
from app.ml.predict import generate_projections


def main() -> None:
    season = int(sys.argv[1]) if len(sys.argv) > 1 else dt.date.today().year
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    db = SessionLocal()
    try:
        print(f"Refreshing consensus ADP for {season}...")
        print(f"  -> {ingest_consensus_adp(db, season)}")

        print(f"Generating projections for {season} week {week}...")
        print(f"  -> {generate_projections(db, season, week)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
