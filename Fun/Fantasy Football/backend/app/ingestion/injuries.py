"""Weekly injury report ingestion. Same cache-once-unless-current-season pattern as weekly
stats (see nfl_data.py's is_current_season) -- injury designations obviously change week to
week during the season and must never go stale.
"""

from __future__ import annotations

import nfl_data_py as nfl
import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Player, PlayerInjuryReport
from app.ingestion.nfl_data import is_cached, is_current_season, mark_cached


def ingest_injuries(db: Session, season: int, force: bool = False) -> int:
    live = is_current_season(season)
    if not force and not live and is_cached(db, season, "injuries"):
        return 0

    try:
        df = nfl.import_injuries([season])
    except Exception:
        if live:
            return 0
        raise

    df = df[df["report_status"].notna() & df["gsis_id"].notna()]
    existing_player_ids = {pid for (pid,) in db.query(Player.id).all()}

    written = 0
    for _, row in df.iterrows():
        player_id = str(row["gsis_id"])
        if player_id not in existing_player_ids:
            continue

        pk = {"player_id": player_id, "season": int(row["season"]), "week": int(row["week"])}
        existing = db.get(PlayerInjuryReport, pk)
        status = str(row["report_status"])
        primary = row.get("report_primary_injury")
        primary = str(primary) if primary and not pd.isna(primary) else None

        if existing:
            existing.report_status = status
            existing.primary_injury = primary
        else:
            db.add(PlayerInjuryReport(**pk, report_status=status, primary_injury=primary))
        written += 1

    db.commit()
    mark_cached(db, season, "injuries")

    if live:
        # Player.injury_status is a "right now" snapshot -- only mirror it from the current
        # season, never from a completed historical season (which would show stale/final-week
        # designations that no longer mean anything).
        _sync_player_injury_status(db, season)
    return written


def _sync_player_injury_status(db: Session, season: int) -> None:
    """Mirrors each player's most recent report_status this season onto Player.injury_status
    for cheap display without a join. Cleared (no current report) resets it to None.
    """
    latest_week = db.query(PlayerInjuryReport.week).filter(PlayerInjuryReport.season == season)
    max_week = latest_week.order_by(PlayerInjuryReport.week.desc()).limit(1).scalar()
    if max_week is None:
        return

    reports = {
        r.player_id: r.report_status
        for r in db.query(PlayerInjuryReport).filter(
            PlayerInjuryReport.season == season, PlayerInjuryReport.week == max_week
        )
    }

    for player in db.query(Player).all():
        player.injury_status = reports.get(player.id)
    db.commit()
