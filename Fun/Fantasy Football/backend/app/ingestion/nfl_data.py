"""Historical (cache-once) ingestion: weekly rosters, weekly stats, schedules for past seasons.

Each (season, data_type) pair is only pulled once and then recorded in the cache manifest.
This module intentionally does NOT own current-season team assignments -- see rosters.py,
which re-pulls those on every run instead of trusting a stale cache, because a player's
current team can change (trade, free agency, release) independent of any of the season's
weekly game data existing yet.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import nfl_data_py as nfl
from sqlalchemy.orm import Session

from app.db.models import IngestionCacheManifest, Player, PlayerWeeklyStat

VALID_POSITIONS = {"QB", "RB", "WR", "TE", "K"}

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
}


def _normalize_team(abbr: str | None) -> str:
    if not abbr or pd.isna(abbr):
        return "FA"
    return TEAM_ALIASES.get(abbr, abbr)


def is_cached(db: Session, season: int, data_type: str) -> bool:
    return db.get(IngestionCacheManifest, {"season": season, "data_type": data_type}) is not None


def mark_cached(db: Session, season: int, data_type: str) -> None:
    existing = db.get(IngestionCacheManifest, {"season": season, "data_type": data_type})
    if existing is None:
        db.add(IngestionCacheManifest(season=season, data_type=data_type))
    else:
        existing.refreshed_at = dt.datetime.utcnow()
    db.commit()


def ingest_historical_rosters(db: Session, season: int, force: bool = False) -> int:
    """Backfills the players table from a past season's weekly rosters. Cache-once.

    This only fills in players who don't already exist -- it must never clobber a player's
    current team/status with historical data (see ingestion/rosters.py for the current,
    always-refreshed team assignment).
    """
    if not force and is_cached(db, season, "rosters"):
        return 0

    rosters_df = nfl.import_weekly_rosters([season])
    rosters_df = rosters_df[rosters_df["position"].isin(VALID_POSITIONS)]
    latest = rosters_df.sort_values("week").groupby("player_id").last().reset_index()

    added = 0
    for _, row in latest.iterrows():
        player_id = str(row["player_id"])
        if db.get(Player, player_id) is not None:
            continue

        name = row.get("player_name")
        if pd.isna(name) or not name:
            first = row.get("first_name", "")
            last = row.get("last_name", "")
            name = f"{first} {last}".strip()

        age = row.get("age")
        exp = row.get("years_exp")

        db.add(
            Player(
                id=player_id,
                name=str(name),
                position=str(row["position"]),
                nfl_team=_normalize_team(row.get("team")),
                status=str(row.get("status")) if not pd.isna(row.get("status")) else "Active",
                age=int(age) if age is not None and not pd.isna(age) else None,
                experience=int(exp) if exp is not None and not pd.isna(exp) else None,
            )
        )
        added += 1

    db.commit()
    mark_cached(db, season, "rosters")
    return added


def is_current_season(season: int) -> bool:
    """Seasons still in progress are never cache-once -- new weeks land every week."""
    return season >= dt.date.today().year


def ingest_historical_stats(db: Session, season: int, force: bool = False) -> int:
    """Pulls weekly actual stats for a season.

    Cache-once for completed past seasons. The current (in-progress) season is always
    re-pulled in full regardless of the manifest -- otherwise, once Week 1 was ingested and
    marked cached, Week 2+ would never be picked up for the rest of the season. Re-pulling a
    full season from nfl_data_py is cheap (it's cached locally by the library itself after the
    first download each run), so this is not worth being clever about at the week level.
    """
    live = is_current_season(season)
    if not force and not live and is_cached(db, season, "weekly_stats"):
        return 0

    try:
        stats_df = nfl.import_weekly_data([season])
    except Exception:
        # nfl_data_py occasionally 404s on the current season before nflverse publishes it;
        # fall back to the raw nflverse-data release parquet directly.
        url = (
            "https://github.com/nflverse/nflverse-data/releases/download/"
            f"stats_player/stats_player_week_{season}.parquet"
        )
        try:
            stats_df = pd.read_parquet(url)
        except Exception:
            if live:
                # Genuinely no games played yet this season (e.g. preseason) -- not an error,
                # just nothing to ingest yet. Don't mark cached, so the next run tries again.
                return 0
            raise

    numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
    stats_df[numeric_cols] = stats_df[numeric_cols].fillna(0.0)

    existing_player_ids = {pid for (pid,) in db.query(Player.id).all()}

    rows_written = 0
    for _, row in stats_df.iterrows():
        player_id = str(row["player_id"])
        if player_id not in existing_player_ids:
            # Stat row for a player not in our players table yet (e.g. a position we filtered
            # out of rosters, like a fullback tagged oddly) -- skip rather than FK-violate.
            continue

        stats = {
            "passing_yards": float(row.get("passing_yards", 0.0)),
            "passing_tds": int(row.get("passing_tds", 0)),
            "interceptions": float(row.get("interceptions", row.get("passing_interceptions", 0.0))),
            "rushing_yards": float(row.get("rushing_yards", 0.0)),
            "rushing_tds": int(row.get("rushing_tds", 0)),
            "receptions": int(row.get("receptions", 0)),
            "receiving_yards": float(row.get("receiving_yards", 0.0)),
            "receiving_tds": int(row.get("receiving_tds", 0)),
            "targets": int(row.get("targets", 0)),
            "fumbles_lost": float(row.get("rushing_fumbles_lost", 0.0)) + float(row.get("receiving_fumbles_lost", 0.0)),
            "two_point_conversions": (
                int(row.get("passing_2pt_conversions", 0))
                + int(row.get("rushing_2pt_conversions", 0))
                + int(row.get("receiving_2pt_conversions", 0))
            ),
        }

        pk = {"player_id": player_id, "season": season, "week": int(row["week"])}
        existing = db.get(PlayerWeeklyStat, pk)
        if existing:
            existing.stats = stats
        else:
            db.add(PlayerWeeklyStat(**pk, stats=stats))
        rows_written += 1

    db.commit()
    mark_cached(db, season, "weekly_stats")
    return rows_written


def ingest_season_range(db: Session, start_year: int, end_year: int, force: bool = False) -> dict:
    summary = {}
    for season in range(start_year, end_year + 1):
        added_players = ingest_historical_rosters(db, season, force=force)
        stat_rows = ingest_historical_stats(db, season, force=force)
        summary[season] = {"players_added": added_players, "stat_rows": stat_rows}
    return summary
