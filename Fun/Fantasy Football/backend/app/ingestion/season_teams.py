"""Season-level team history, powering the ML pipeline's new-team flag.

Historical seasons are cache-once (re-pulling nfl_data_py's weekly rosters is cheap since it's
locally cached by the library after first download, but there's no reason to redo the groupby
every run). The current season is always written fresh from the Player table, which itself is
kept live by ingestion/rosters.py.
"""

from __future__ import annotations

import datetime as dt

import nfl_data_py as nfl
from sqlalchemy.orm import Session

from app.db.models import IngestionCacheManifest, Player, PlayerSeasonTeam
from app.ingestion.nfl_data import VALID_POSITIONS, _normalize_team


def backfill_historical_season_teams(db: Session, season: int, force: bool = False) -> int:
    manifest_key = {"season": season, "data_type": "season_team"}
    if not force and db.get(IngestionCacheManifest, manifest_key) is not None:
        return 0

    rosters_df = nfl.import_weekly_rosters([season])
    rosters_df = rosters_df[rosters_df["position"].isin(VALID_POSITIONS)]
    latest = rosters_df.sort_values("week").groupby("player_id").last().reset_index()

    written = 0
    for _, row in latest.iterrows():
        player_id = str(row["player_id"])
        team = _normalize_team(row.get("team"))
        pk = {"player_id": player_id, "season": season}
        existing = db.get(PlayerSeasonTeam, pk)
        if existing:
            existing.nfl_team = team
        else:
            db.add(PlayerSeasonTeam(**pk, nfl_team=team))
        written += 1

    db.add(IngestionCacheManifest(**manifest_key))
    db.commit()
    return written


def write_current_season_team(db: Session) -> int:
    """Always-fresh: mirrors the live-refreshed Player.nfl_team into the current season row."""
    current_year = dt.date.today().year
    players = db.query(Player).all()

    written = 0
    for p in players:
        pk = {"player_id": p.id, "season": current_year}
        existing = db.get(PlayerSeasonTeam, pk)
        if existing:
            existing.nfl_team = p.nfl_team
        else:
            db.add(PlayerSeasonTeam(**pk, nfl_team=p.nfl_team))
        written += 1

    db.commit()
    return written
