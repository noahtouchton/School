"""Yahoo Fantasy integration endpoints: OAuth, league browsing, the live draft
assistant, lineup optimization, and waiver/transaction execution.

Anything that hits Yahoo repeatedly during a 5-second draft poll (settings,
teams, resolved player keys, the local draft board) is cached in-process with
a short TTL so a live draft doesn't hammer either Yahoo or our DB.
"""

from __future__ import annotations

import csv
import io
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.draft import assistant, lineup as lineup_mod
from app.yahoo import client, oauth

router = APIRouter(prefix="/yahoo", tags=["yahoo"])


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except oauth.YahooAuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except client.YahooApiError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ---------------------------------------------------------------------------
# Tiny TTL cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, ttl: float, producer):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = producer()
    _cache[key] = (now, value)
    return value


# Resolved player keys never change; keep them for the whole draft.
_player_key_cache: dict[str, dict] = {}


def _resolve_players(player_keys: list[str]) -> dict[str, dict]:
    missing = [k for k in player_keys if k not in _player_key_cache]
    if missing:
        for p in _guard(client.get_players_by_keys, missing):
            if p.get("player_key"):
                _player_key_cache[p["player_key"]] = p
    return {k: _player_key_cache[k] for k in player_keys if k in _player_key_cache}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class CodePayload(BaseModel):
    code: str


@router.get("/status")
def status() -> dict:
    settings = get_settings()
    return {
        "has_credentials": bool(settings.yahoo_client_id and settings.yahoo_client_secret),
        "connected": oauth.is_connected(),
    }


@router.get("/auth/login")
def auth_login() -> dict:
    return {"authorize_url": _guard(oauth.authorize_url)}


@router.get("/auth/callback")
def auth_callback(code: str | None = None, error: str | None = None):
    settings = get_settings()
    if error or not code:
        return RedirectResponse(f"{settings.frontend_url}/yahoo?error={error or 'no_code'}")
    _guard(oauth.exchange_code, code)
    return RedirectResponse(f"{settings.frontend_url}/yahoo?connected=1")


@router.post("/auth/code")
def auth_code(payload: CodePayload) -> dict:
    _guard(oauth.exchange_code, payload.code)
    return {"connected": True}


@router.post("/auth/disconnect")
def auth_disconnect() -> dict:
    oauth.clear_tokens()
    _cache.clear()
    return {"connected": False}


# ---------------------------------------------------------------------------
# Leagues
# ---------------------------------------------------------------------------

@router.get("/leagues")
def leagues() -> dict:
    return {"leagues": _guard(client.get_user_leagues)}


def _league_settings(league_key: str) -> dict:
    return _cached(f"settings:{league_key}", 300, lambda: _guard(client.get_league_settings, league_key))


def _league_teams(league_key: str) -> list[dict]:
    return _cached(f"teams:{league_key}", 120, lambda: _guard(client.get_league_teams, league_key))


def _my_team(league_key: str) -> dict:
    teams = _league_teams(league_key)
    mine = next((t for t in teams if t["is_mine"]), None)
    if mine is None:
        raise HTTPException(status_code=404, detail="No team owned by you in this league")
    return mine


@router.get("/leagues/{league_key}")
def league_detail(league_key: str) -> dict:
    settings = _league_settings(league_key)
    teams = _league_teams(league_key)
    mine = next((t for t in teams if t["is_mine"]), None)
    return {"league": settings, "teams": teams, "my_team_key": mine["team_key"] if mine else None}


# ---------------------------------------------------------------------------
# Draft board / cheat sheet
# ---------------------------------------------------------------------------

def _board(db: Session, league_key: str) -> list[assistant.BoardPlayer]:
    def produce():
        board = assistant.build_board(db)
        settings = _league_settings(league_key)
        assistant.apply_vorp(
            board, settings.get("roster_positions", []), settings.get("num_teams", 12) or 12
        )
        return board

    return _cached(f"board:{league_key}", 300, produce)


def _board_entry(bp: assistant.BoardPlayer) -> dict:
    return {
        "name": bp.name,
        "position": bp.position,
        "nfl_team": bp.nfl_team,
        "season_points": round(bp.season_points, 1),
        "per_game": round(bp.per_game, 2) if bp.per_game is not None else None,
        "vorp": bp.vorp,
        "tier": bp.tier,
        "overall_rank": bp.overall_rank,
        "position_rank": bp.position_rank,
        "adp": bp.adp,
        "source": bp.source,
        "injury_status": bp.injury_status,
    }


@router.get("/leagues/{league_key}/cheatsheet")
def cheatsheet(
    league_key: str,
    limit: int = Query(default=250, le=600),
    db: Session = Depends(get_db),
) -> dict:
    board = _board(db, league_key)
    return {"players": [_board_entry(bp) for bp in board[:limit]]}


@router.get("/leagues/{league_key}/cheatsheet.csv", response_class=PlainTextResponse)
def cheatsheet_csv(league_key: str, db: Session = Depends(get_db)) -> str:
    """CSV export — mirror this order into Yahoo's Pre-Draft Rankings and even an
    autodraft will follow the AI board."""
    board = _board(db, league_key)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["rank", "name", "position", "team", "season_points", "vorp", "tier", "adp"])
    for bp in board[:400]:
        writer.writerow(
            [bp.overall_rank, bp.name, bp.position, bp.nfl_team, f"{bp.season_points:.1f}", bp.vorp, bp.tier, bp.adp or ""]
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Live draft
# ---------------------------------------------------------------------------

def _snake_next_pick(num_teams: int, draft_position: int, current_pick: int) -> int | None:
    """The next overall pick number belonging to draft_position at/after current_pick."""
    if not num_teams or not draft_position:
        return None
    for rnd in range(1, 40):
        slot = draft_position if rnd % 2 == 1 else num_teams - draft_position + 1
        overall = (rnd - 1) * num_teams + slot
        if overall >= current_pick:
            return overall
    return None


@router.get("/leagues/{league_key}/draft")
def draft_state(league_key: str, db: Session = Depends(get_db)) -> dict:
    settings = _league_settings(league_key)
    teams = _league_teams(league_key)
    team_by_key = {t["team_key"]: t for t in teams}
    mine = next((t for t in teams if t["is_mine"]), None)
    my_team_key = mine["team_key"] if mine else None

    picks = _guard(client.get_draft_results, league_key)
    resolved = _resolve_players([p["player_key"] for p in picks])

    pick_rows = []
    my_players: list[dict] = []
    for p in picks:
        player = resolved.get(p["player_key"], {})
        row = {
            **p,
            "team_name": team_by_key.get(p["team_key"], {}).get("name"),
            "player": {
                "name": player.get("name"),
                "position": player.get("primary_position") or player.get("position"),
                "nfl_team": player.get("nfl_team"),
            },
        }
        pick_rows.append(row)
        if p["team_key"] == my_team_key:
            my_players.append(player)

    num_teams = settings.get("num_teams", 0) or len(teams)
    current_pick = len(picks) + 1
    total_rounds = sum(
        s.get("count", 0)
        for s in settings.get("roster_positions", [])
        if s.get("position") not in ("IR", "IR+")
    )

    draft_position = None
    if mine and mine.get("draft_position"):
        try:
            draft_position = int(mine["draft_position"])
        except (TypeError, ValueError):
            draft_position = None
    if draft_position is None and my_team_key:
        # Infer from my first observed pick.
        my_first = next((p for p in picks if p["team_key"] == my_team_key), None)
        if my_first:
            slot = ((my_first["pick"] - 1) % num_teams) + 1
            draft_position = slot if my_first["round"] % 2 == 1 else num_teams - slot + 1

    my_next_pick = (
        _snake_next_pick(num_teams, draft_position, current_pick) if draft_position else None
    )
    picks_until = (my_next_pick - current_pick) if my_next_pick else num_teams

    on_clock_key = None
    if num_teams and current_pick <= num_teams * total_rounds:
        rnd = (current_pick - 1) // num_teams + 1
        slot = ((current_pick - 1) % num_teams) + 1
        clock_pos = slot if rnd % 2 == 1 else num_teams - slot + 1
        on_clock_key = next(
            (
                t["team_key"]
                for t in teams
                if str(t.get("draft_position") or "") == str(clock_pos)
            ),
            None,
        )

    board = _board(db, league_key)
    taken, taken_dst = assistant.match_taken(board, list(resolved.values()))
    my_positions = [
        (p.get("primary_position") or p.get("position") or "") for p in my_players
    ]
    my_positions = ["DST" if p == "DEF" else p for p in my_positions]

    recommendations = assistant.recommend(
        board,
        taken,
        taken_dst,
        my_positions,
        settings.get("roster_positions", []),
        picks_until_next=picks_until,
        top_n=8,
    )

    best_available = [
        _board_entry(bp)
        for bp in board
        if (bp.norm_name, bp.position) not in taken
        and not (bp.position == "DST" and (bp.nfl_team or "") in taken_dst)
    ][:40]

    return {
        "draft_status": settings.get("draft_status"),
        "draft_type": settings.get("draft_type"),
        "draft_time": settings.get("draft_time"),
        "num_teams": num_teams,
        "total_rounds": total_rounds,
        "my_team_key": my_team_key,
        "my_draft_position": draft_position,
        "current_pick": current_pick,
        "my_next_pick": my_next_pick,
        "picks_until_my_turn": picks_until,
        "on_the_clock": team_by_key.get(on_clock_key, {}).get("name") if on_clock_key else None,
        "i_am_on_the_clock": bool(on_clock_key and on_clock_key == my_team_key),
        "picks": pick_rows,
        "my_roster": [
            {
                "name": p.get("name"),
                "position": p.get("primary_position") or p.get("position"),
                "nfl_team": p.get("nfl_team"),
            }
            for p in my_players
        ],
        "recommendations": recommendations,
        "best_available": best_available,
    }


# ---------------------------------------------------------------------------
# My team: roster, lineup, waivers, transactions
# ---------------------------------------------------------------------------

@router.get("/leagues/{league_key}/team")
def my_team(
    league_key: str,
    week: int | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict:
    settings = _league_settings(league_key)
    mine = _my_team(league_key)
    roster = _guard(client.get_team_roster, mine["team_key"], week)
    lookup, proj_season, proj_week = lineup_mod.projection_lookup(db)
    optimal = lineup_mod.optimal_lineup(roster, settings.get("roster_positions", []), lookup)
    return {
        "team": mine,
        "league": {k: settings.get(k) for k in ("name", "season", "current_week", "num_teams", "uses_faab")},
        "projection_week": {"season": proj_season, "week": proj_week},
        "lineup": optimal,
    }


class ApplyLineupPayload(BaseModel):
    week: int


@router.post("/leagues/{league_key}/lineup")
def apply_lineup(league_key: str, payload: ApplyLineupPayload, db: Session = Depends(get_db)) -> dict:
    settings = _league_settings(league_key)
    mine = _my_team(league_key)
    roster = _guard(client.get_team_roster, mine["team_key"], payload.week)
    lookup, _, _ = lineup_mod.projection_lookup(db)
    optimal = lineup_mod.optimal_lineup(roster, settings.get("roster_positions", []), lookup)
    if not optimal["changes"]:
        return {"applied": False, "changes": [], "detail": "Lineup is already optimal"}
    _guard(
        client.set_lineup,
        mine["team_key"],
        payload.week,
        lineup_mod.lineup_change_payload(optimal),
    )
    return {"applied": True, "changes": optimal["changes"], "improvement": optimal["improvement"]}


@router.get("/leagues/{league_key}/waivers")
def waivers(league_key: str, db: Session = Depends(get_db)) -> dict:
    mine = _my_team(league_key)
    roster = _guard(client.get_team_roster, mine["team_key"])
    lookup, _, _ = lineup_mod.projection_lookup(db)

    free_agents: list[dict] = []
    for pos in ("QB", "RB", "WR", "TE"):
        free_agents.extend(
            _cached(
                f"fa:{league_key}:{pos}",
                120,
                lambda pos=pos: _guard(client.get_league_players, league_key, "A", pos, 25),
            )
        )
    # "A" (all available) includes both free agents and players on waivers.
    available = [
        fa
        for fa in free_agents
        if fa.get("ownership_type") in (None, "freeagents", "waivers")
    ]
    recs = lineup_mod.waiver_recommendations(available, roster, lookup)
    return {"faab_balance": mine.get("faab_balance"), "recommendations": recs[:20]}


class TransactionPayload(BaseModel):
    add_player_key: str | None = None
    drop_player_key: str | None = None
    faab_bid: int | None = None


@router.post("/leagues/{league_key}/transactions")
def execute_transaction(league_key: str, payload: TransactionPayload) -> dict:
    if not payload.add_player_key and not payload.drop_player_key:
        raise HTTPException(status_code=422, detail="Provide add_player_key and/or drop_player_key")
    mine = _my_team(league_key)
    _guard(
        client.execute_transaction,
        league_key,
        mine["team_key"],
        payload.add_player_key,
        payload.drop_player_key,
        payload.faab_bid,
    )
    _cache.pop(f"teams:{league_key}", None)
    return {"ok": True}
