"""Current-season team/roster assignment -- ALWAYS refreshed, never treated as cache-once.

This is the fix for the old app's stale-team-data bug (see plan: a player who signs with a
new team should show the new team immediately, not whatever their last cached weekly-roster
snapshot said, and not require a hand patch in the DB). `nfl_data_py.import_players()` pulls
nflverse's continuously-updated player master file, which reflects free agency/trades/releases
independent of whether any games have been played yet this season -- unlike weekly rosters,
which only update once real game weeks exist.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import nfl_data_py as nfl
from sqlalchemy.orm import Session

from app.db.models import IngestionCacheManifest, Player
from app.ingestion.nfl_data import VALID_POSITIONS, _normalize_team

ALL_NFL_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB",
    "TEN", "WAS",
}


def _age_from_birth_date(birth_date: str | None) -> int | None:
    if not birth_date or pd.isna(birth_date):
        return None
    try:
        born = dt.date.fromisoformat(str(birth_date))
    except ValueError:
        return None
    today = dt.date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def refresh_current_rosters(db: Session) -> dict:
    """Upserts every fantasy-relevant player's current team/status/age/experience.

    Runs unconditionally on every ingestion pass -- intentionally not gated by the cache
    manifest the way historical season data is.
    """
    players_df = nfl.import_players()
    players_df = players_df[players_df["position"].isin(VALID_POSITIONS)]
    # Only players active in the last 2 seasons are fantasy-relevant; this also excludes
    # long-retired players who happen to share a position filter.
    current_year = dt.date.today().year
    players_df = players_df[players_df["last_season"].fillna(0) >= current_year - 1]

    updated = 0
    created = 0
    for _, row in players_df.iterrows():
        player_id = row.get("gsis_id")
        if not player_id or pd.isna(player_id):
            continue
        player_id = str(player_id)

        name = row.get("display_name")
        if not name or pd.isna(name):
            continue

        team = _normalize_team(row.get("latest_team"))
        status = row.get("status")
        status = str(status) if status and not pd.isna(status) else "Active"
        exp = row.get("years_of_experience")

        player = db.get(Player, player_id)
        if player is None:
            db.add(
                Player(
                    id=player_id,
                    name=str(name),
                    position=str(row["position"]),
                    nfl_team=team,
                    status=status,
                    age=_age_from_birth_date(row.get("birth_date")),
                    experience=int(exp) if exp is not None and not pd.isna(exp) else None,
                )
            )
            created += 1
        else:
            player.name = str(name)
            player.position = str(row["position"])
            player.nfl_team = team
            player.status = status
            age = _age_from_birth_date(row.get("birth_date"))
            if age is not None:
                player.age = age
            if exp is not None and not pd.isna(exp):
                player.experience = int(exp)
            updated += 1

    for team_abbr in ALL_NFL_TEAMS:
        dst_id = f"DST_{team_abbr}"
        dst = db.get(Player, dst_id)
        if dst is None:
            db.add(Player(id=dst_id, name=f"{team_abbr} Defense", position="DST", nfl_team=team_abbr, status="Active"))
            created += 1

    db.commit()

    manifest_row = db.get(IngestionCacheManifest, {"season": current_year, "data_type": "current_roster"})
    if manifest_row is None:
        db.add(IngestionCacheManifest(season=current_year, data_type="current_roster"))
    else:
        manifest_row.refreshed_at = dt.datetime.utcnow()
    db.commit()

    return {"created": created, "updated": updated}
