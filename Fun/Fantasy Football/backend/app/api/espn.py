"""ESPN Fantasy endpoints: connect a league, live draft assistant, lineup
optimization, waivers, and transactions.

Connection details (league id, year, cookies) persist in the espn_league_links
table so the link survives restarts. Cookies are stored locally in your own
SQLite file and never leave this machine except in requests to ESPN itself.
"""

from __future__ import annotations

import csv
import io
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import EspnLeagueLink
from app.db.session import get_db
from app.draft import assistant, lineup as lineup_mod
from app.espn import client

router = APIRouter(prefix="/espn", tags=["espn"])

LINK_ID = "primary"


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except client.EspnError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------

class ConnectPayload(BaseModel):
    league_id: int
    year: int
    espn_s2: str | None = None
    swid: str | None = None
    nickname: str | None = None


def _stored_link(db: Session) -> EspnLeagueLink | None:
    link = db.get(EspnLeagueLink, LINK_ID)
    if link is not None:
        return link
    # Fall back to .env so a league configured there works without any UI step.
    settings = get_settings()
    if settings.espn_league_id:
        return EspnLeagueLink(
            id=LINK_ID,
            league_id=settings.espn_league_id,
            year=settings.espn_year,
            espn_s2=settings.espn_s2,
            swid=settings.espn_swid,
            nickname=None,
        )
    return None


def _require_link(db: Session) -> EspnLeagueLink:
    link = _stored_link(db)
    if link is None:
        raise HTTPException(
            status_code=400,
            detail="No ESPN league connected. Add your league id, year and cookies first.",
        )
    return link


# League objects are expensive to construct (several ESPN round trips), so a
# short-lived cache keeps a 5-second draft poll from rebuilding one each time.
_league_cache: dict[tuple, tuple[float, object]] = {}
_LEAGUE_TTL = 20


def _league(link: EspnLeagueLink, fresh: bool = False):
    key = (link.league_id, link.year)
    hit = _league_cache.get(key)
    if hit and not fresh and time.time() - hit[0] < _LEAGUE_TTL:
        return hit[1]
    league = _guard(client.connect, link.league_id, link.year, link.espn_s2, link.swid)
    _league_cache[key] = (time.time(), league)
    return league


def _my_team_or_404(league, link: EspnLeagueLink):
    team = client.my_team(league, link.swid)
    if team is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Couldn't tell which team is yours. That usually means the SWID cookie "
                "doesn't match a manager in this league -- check you copied it from the "
                "account that owns the team."
            ),
        )
    return team


@router.get("/status")
def status(db: Session = Depends(get_db)) -> dict:
    link = _stored_link(db)
    if link is None:
        return {"connected": False}
    return {
        "connected": True,
        "league_id": link.league_id,
        "year": link.year,
        "nickname": link.nickname,
        "has_cookies": bool(link.espn_s2 and link.swid),
    }


@router.post("/connect")
def connect(payload: ConnectPayload, db: Session = Depends(get_db)) -> dict:
    # Validate before persisting so a bad cookie never gets saved as "connected".
    league = _guard(
        client.connect, payload.league_id, payload.year, payload.espn_s2, payload.swid
    )
    team = client.my_team(league, payload.swid)

    db.merge(
        EspnLeagueLink(
            id=LINK_ID,
            league_id=payload.league_id,
            year=payload.year,
            espn_s2=payload.espn_s2,
            swid=payload.swid,
            nickname=payload.nickname,
        )
    )
    db.commit()
    _league_cache.clear()

    return {
        "connected": True,
        "league_name": getattr(league.settings, "name", None),
        "num_teams": len(league.teams),
        "current_week": league.current_week,
        "my_team": getattr(team, "team_name", None),
        "identified_my_team": team is not None,
        "teams": [{"team_id": t.team_id, "name": t.team_name} for t in league.teams],
    }


@router.post("/disconnect")
def disconnect(db: Session = Depends(get_db)) -> dict:
    link = db.get(EspnLeagueLink, LINK_ID)
    if link is not None:
        db.delete(link)
        db.commit()
    _league_cache.clear()
    return {"connected": False}


@router.get("/league")
def league_detail(db: Session = Depends(get_db)) -> dict:
    link = _require_link(db)
    league = _league(link)
    team = client.my_team(league, link.swid)
    return {
        "league_id": link.league_id,
        "year": link.year,
        "name": getattr(league.settings, "name", None),
        "num_teams": len(league.teams),
        "current_week": league.current_week,
        "roster_positions": client.roster_positions(league),
        "my_team_id": getattr(team, "team_id", None),
        "my_team_name": getattr(team, "team_name", None),
        "teams": [
            {"team_id": t.team_id, "name": t.team_name, "owners": len(getattr(t, "owners", []) or [])}
            for t in league.teams
        ],
    }


# ---------------------------------------------------------------------------
# Draft board
# ---------------------------------------------------------------------------

_board_cache: dict[tuple, tuple[float, list]] = {}
_BOARD_TTL = 600


def _board(db: Session, slots: list[dict], num_teams: int) -> list[assistant.BoardPlayer]:
    key = (num_teams, tuple((s["position"], s["count"]) for s in slots))
    hit = _board_cache.get(key)
    if hit and time.time() - hit[0] < _BOARD_TTL:
        return hit[1]
    board = assistant.build_board(db)
    assistant.apply_vorp(board, slots, num_teams)
    _board_cache[key] = (time.time(), board)
    return board


def _entry(bp: assistant.BoardPlayer) -> dict:
    return {
        "name": bp.name,
        "position": bp.position,
        "nfl_team": bp.nfl_team,
        "season_points": round(bp.season_points, 1),
        "vorp": bp.vorp,
        "tier": bp.tier,
        "overall_rank": bp.overall_rank,
        "position_rank": bp.position_rank,
        "adp": bp.adp,
        "source": bp.source,
        "injury_status": bp.injury_status,
    }


@router.get("/draft")
def draft_state(db: Session = Depends(get_db)) -> dict:
    link = _require_link(db)
    league = _league(link, fresh=True)
    slots = client.roster_positions(league)
    num_teams = len(league.teams)
    team = client.my_team(league, link.swid)
    my_team_id = getattr(team, "team_id", None)

    picks = _guard(client.draft_picks, league)
    my_players = [p for p in picks if p["team_id"] == my_team_id]

    board = _board(db, slots, num_teams)
    taken, taken_dst = assistant.match_taken(
        board,
        [{"name": p["name"], "primary_position": "", "nfl_team": ""} for p in picks],
    )

    # Resolve my drafted players to board positions for accurate roster needs.
    mine_norm = {assistant.normalize_name(p["name"]) for p in my_players}
    my_positions = [bp.position for bp in board if bp.norm_name in mine_norm]

    total_rounds = sum(
        s["count"] for s in slots if s["position"] not in ("IR", "IR+")
    )
    current_pick = len(picks) + 1
    drafted_by_me = len(my_players)
    # Snake order: my next pick is roughly num_teams away, refined once we know
    # my slot from the first round's pick order.
    my_slot = None
    first_round = [p for p in picks if p["round"] == 1]
    for p in first_round:
        if p["team_id"] == my_team_id:
            my_slot = p["round_pick"]
            break

    picks_until = num_teams
    my_next_pick = None
    if my_slot:
        for rnd in range(1, total_rounds + 2):
            slot = my_slot if rnd % 2 == 1 else num_teams - my_slot + 1
            overall = (rnd - 1) * num_teams + slot
            if overall >= current_pick:
                my_next_pick = overall
                picks_until = overall - current_pick
                break

    recommendations = assistant.recommend(
        board,
        taken,
        taken_dst,
        my_positions,
        slots,
        picks_until_next=picks_until,
        top_n=8,
    )
    available = [
        bp
        for bp in board
        if (bp.norm_name, bp.position) not in taken
        and not (bp.position == "DST" and (bp.nfl_team or "") in taken_dst)
    ]

    return {
        "drafted": len(picks),
        "current_pick": current_pick,
        "total_rounds": total_rounds,
        "num_teams": num_teams,
        "my_team_id": my_team_id,
        "my_team_name": getattr(team, "team_name", None),
        "my_draft_slot": my_slot,
        "my_next_pick": my_next_pick,
        "picks_until_my_turn": picks_until,
        "drafted_by_me": drafted_by_me,
        "picks": [
            {
                "round": p["round"],
                "round_pick": p["round_pick"],
                "name": p["name"],
                "team_name": p["team_name"],
                "is_mine": p["team_id"] == my_team_id,
            }
            for p in picks
        ],
        "my_roster": [{"name": p["name"]} for p in my_players],
        "my_roster_positions": my_positions,
        "recommendations": recommendations,
        "best_available": [_entry(bp) for bp in available[:40]],
    }


@router.get("/cheatsheet.csv", response_class=PlainTextResponse)
def cheatsheet_csv(db: Session = Depends(get_db)) -> str:
    link = _require_link(db)
    league = _league(link)
    board = _board(db, client.roster_positions(league), len(league.teams))
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["rank", "name", "position", "team", "season_points", "vorp", "tier", "adp", "source"])
    for bp in board[:400]:
        writer.writerow(
            [bp.overall_rank, bp.name, bp.position, bp.nfl_team, f"{bp.season_points:.1f}",
             bp.vorp, bp.tier, bp.adp or "", bp.source]
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# My team: lineup + waivers
# ---------------------------------------------------------------------------

@router.get("/team")
def my_team(db: Session = Depends(get_db)) -> dict:
    link = _require_link(db)
    league = _league(link)
    team = _my_team_or_404(league, link)
    roster = client.team_roster(league, team)
    slots = client.roster_positions(league)

    lookup, proj_season, proj_week = lineup_mod.projection_lookup(db)
    optimal = lineup_mod.optimal_lineup(roster, slots, lookup)

    return {
        "team": {
            "team_id": team.team_id,
            "name": team.team_name,
            "wins": team.wins,
            "losses": team.losses,
            "points_for": team.points_for,
            "acquisition_budget_spent": getattr(team, "acquisition_budget_spent", 0),
        },
        "league": {
            "name": getattr(league.settings, "name", None),
            "year": link.year,
            "current_week": league.current_week,
            "num_teams": len(league.teams),
        },
        "projection_week": {"season": proj_season, "week": proj_week},
        "lineup": optimal,
    }


class ApplyLineupPayload(BaseModel):
    week: int | None = None


@router.post("/lineup")
def apply_lineup(payload: ApplyLineupPayload, db: Session = Depends(get_db)) -> dict:
    link = _require_link(db)
    league = _league(link, fresh=True)
    team = _my_team_or_404(league, link)
    slots = client.roster_positions(league)
    week = payload.week or league.current_week

    roster = client.team_roster(league, team)
    lookup, _, _ = lineup_mod.projection_lookup(db)
    optimal = lineup_mod.optimal_lineup(roster, slots, lookup)
    if not optimal["changes"]:
        return {"applied": False, "changes": [], "detail": "Lineup is already optimal"}

    by_key = {p["player_key"]: p for p in roster}
    moves = [
        {
            "player_id": int(change["player_key"]),
            "from_slot": by_key[change["player_key"]]["selected_position"] or "BN",
            "to_slot": change["to"],
        }
        for change in optimal["changes"]
    ]
    _guard(
        client.set_lineup,
        league,
        link.league_id,
        link.year,
        link.swid,
        team.team_id,
        week,
        moves,
    )

    # Verify against a fresh read rather than trusting the write's 200.
    _league_cache.clear()
    verify_league = _league(link, fresh=True)
    verify_team = _my_team_or_404(verify_league, link)
    after = lineup_mod.optimal_lineup(
        client.team_roster(verify_league, verify_team), slots, lookup
    )
    return {
        "applied": True,
        "changes": optimal["changes"],
        "improvement": optimal["improvement"],
        "remaining_changes": after["changes"],
        "verified": len(after["changes"]) == 0,
    }


@router.get("/waivers")
def waivers(
    limit: int = Query(default=20, le=50),
    db: Session = Depends(get_db),
) -> dict:
    link = _require_link(db)
    league = _league(link)
    team = _my_team_or_404(league, link)
    roster = client.team_roster(league, team)
    lookup, _, _ = lineup_mod.projection_lookup(db)

    agents: list[dict] = []
    for position in ("QB", "RB", "WR", "TE"):
        agents.extend(_guard(client.free_agents, league, 25, position))

    recs = lineup_mod.waiver_recommendations(agents, roster, lookup)
    return {
        "budget_spent": getattr(team, "acquisition_budget_spent", 0),
        "recommendations": recs[:limit],
    }


class TransactionPayload(BaseModel):
    add_player_id: int | None = None
    drop_player_id: int | None = None
    bid: int | None = None
    week: int | None = None


@router.post("/transactions")
def transaction(payload: TransactionPayload, db: Session = Depends(get_db)) -> dict:
    if payload.add_player_id is None and payload.drop_player_id is None:
        raise HTTPException(status_code=422, detail="Provide add_player_id and/or drop_player_id")
    link = _require_link(db)
    league = _league(link, fresh=True)
    team = _my_team_or_404(league, link)
    week = payload.week or league.current_week

    _guard(
        client.add_drop,
        league,
        link.league_id,
        link.year,
        link.swid,
        team.team_id,
        week,
        payload.add_player_id,
        payload.drop_player_id,
        payload.bid,
    )
    _league_cache.clear()
    return {"ok": True}
