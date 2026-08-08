"""Pulls consensus average-draft-position data from FantasyFootballCalculator's free public API.

This is a calibration/sanity-check signal only (see plan failure mode #2) -- it is never copied
into our own projections, just displayed alongside them so an obviously-wrong model output (the
kind the old app produced) is easy to spot.
"""

from __future__ import annotations

import re

import requests
from sqlalchemy.orm import Session

from app.db.models import ConsensusAdp, Player
from app.ingestion.nfl_data import _normalize_team

FFC_POSITION_MAP = {"PK": "K", "DEF": "DST"}

FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{scoring}?teams=12&year={year}"

_SUFFIX_RE = re.compile(r"\s+(jr\.?|sr\.?|i{2,3}|iv)$", re.IGNORECASE)


def _strip_suffix(name: str) -> str:
    return _SUFFIX_RE.sub("", name).strip()


def _match_player(db: Session, name: str, position: str, team: str) -> str | None:
    if position == "DST":
        return f"DST_{team}"

    search_name = _strip_suffix(name)
    candidates = (
        db.query(Player)
        .filter(Player.position == position)
        .filter(Player.name.ilike(search_name) | Player.name.ilike(f"{search_name}%"))
        .all()
    )
    if len(candidates) == 1:
        return candidates[0].id
    if len(candidates) > 1:
        team_match = [c for c in candidates if c.nfl_team == team]
        if len(team_match) == 1:
            return team_match[0].id
    return None


def ingest_consensus_adp(db: Session, season: int, scoring: str = "half-ppr") -> dict:
    resp = requests.get(FFC_URL.format(scoring=scoring, year=season), timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != "Success":
        return {"status": "no_data", "detail": payload.get("errors")}

    db.query(ConsensusAdp).filter(
        ConsensusAdp.season == season,
        ConsensusAdp.source == "ffc",
        ConsensusAdp.scoring_format == scoring,
    ).delete()

    matched = 0
    total = 0
    for entry in payload["players"]:
        position = FFC_POSITION_MAP.get(entry["position"], entry["position"])
        team = _normalize_team(entry.get("team"))
        player_id = _match_player(db, entry["name"], position, team)
        if player_id:
            matched += 1

        db.add(
            ConsensusAdp(
                season=season,
                source="ffc",
                scoring_format=scoring,
                player_id=player_id,
                name_raw=entry["name"],
                position=position,
                nfl_team=team,
                adp=entry["adp"],
            )
        )
        total += 1

    db.commit()
    return {"status": "ok", "total": total, "matched": matched}
