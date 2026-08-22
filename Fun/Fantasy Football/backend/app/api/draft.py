"""Yahoo-free draft assistant.

Yahoo gates Fantasy API access behind a manual approval process, so the live
Draft Room can sit unusable on the one night it matters. These endpoints run
the exact same valuation and recommendation engine off a draft board the user
maintains by hand (click each player as he comes off the board), which means
draft night never depends on an API approval, a login, or a network round trip
to Yahoo.
"""

from __future__ import annotations

import csv
import io
import time

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.draft import assistant

router = APIRouter(prefix="/draft", tags=["draft"])

# Yahoo's default NFL roster, which is what most leagues actually use.
DEFAULT_ROSTER_POSITIONS = [
    {"position": "QB", "count": 1},
    {"position": "RB", "count": 2},
    {"position": "WR", "count": 2},
    {"position": "TE", "count": 1},
    {"position": "W/R/T", "count": 1},
    {"position": "K", "count": 1},
    {"position": "DEF", "count": 1},
    {"position": "BN", "count": 6},
]


class RosterSlot(BaseModel):
    position: str
    count: int


class BoardRequest(BaseModel):
    num_teams: int = 12
    roster_positions: list[RosterSlot] | None = None
    drafted: list[str] = Field(default_factory=list, description="Names already off the board")
    my_players: list[str] = Field(default_factory=list, description="Names on MY roster")
    picks_until_next: int = 0
    top_n: int = 8

    def slots(self) -> list[dict]:
        if self.roster_positions:
            return [s.model_dump() for s in self.roster_positions]
        return DEFAULT_ROSTER_POSITIONS


# The board is expensive to build (full stat history + ADP curve fitting) and
# identical for every request with the same league shape, so it's cached for
# the length of a draft rather than rebuilt on each pick.
_board_cache: dict[tuple, tuple[float, list[assistant.BoardPlayer]]] = {}
_BOARD_TTL = 900


def _board(db: Session, num_teams: int, slots: list[dict]) -> list[assistant.BoardPlayer]:
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
        "per_game": round(bp.per_game, 2) if bp.per_game is not None else None,
        "vorp": bp.vorp,
        "tier": bp.tier,
        "overall_rank": bp.overall_rank,
        "position_rank": bp.position_rank,
        "adp": bp.adp,
        "source": bp.source,
        "injury_status": bp.injury_status,
    }


def _taken_keys(names: list[str]) -> tuple[set[tuple[str, str]], set[str]]:
    """Mark a name as taken at every position, since the UI tracks names only."""
    taken: set[tuple[str, str]] = set()
    for raw in names:
        norm = assistant.normalize_name(raw)
        if not norm:
            continue
        for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
            taken.add((norm, pos))
    return taken, set()


@router.post("/recommendations")
def recommendations(payload: BoardRequest, db: Session = Depends(get_db)) -> dict:
    slots = payload.slots()
    board = _board(db, payload.num_teams, slots)

    # Anyone on my roster is off the board too, whether or not the caller also
    # listed them in `drafted` -- otherwise the engine happily recommends a
    # player I already own.
    taken, taken_dst = _taken_keys(payload.drafted + payload.my_players)

    # Resolve my players to their board positions so roster needs are accurate.
    mine_norm = {assistant.normalize_name(n) for n in payload.my_players}
    my_positions = [bp.position for bp in board if bp.norm_name in mine_norm]

    picks_until = payload.picks_until_next or payload.num_teams
    recs = assistant.recommend(
        board,
        taken,
        taken_dst,
        my_positions,
        slots,
        picks_until_next=picks_until,
        top_n=payload.top_n,
    )
    available = [bp for bp in board if (bp.norm_name, bp.position) not in taken]
    needs = assistant.compute_needs(slots, my_positions)

    return {
        "recommendations": recs,
        "best_available": [_entry(bp) for bp in available[:50]],
        "my_roster_positions": my_positions,
        "needs": {
            "starters_needed": needs.starters_needed,
            "flex_needed": needs.flex_needed,
            "bench_open": needs.bench_open,
            "total_open": needs.total_open,
        },
        "board_size": len(board),
        "drafted_count": len(payload.drafted),
    }


@router.post("/cheatsheet.csv", response_class=PlainTextResponse)
def cheatsheet_csv(payload: BoardRequest, db: Session = Depends(get_db)) -> str:
    """Full ranked board as CSV. Paste into Yahoo's Pre-Draft Rankings so even
    an autodraft follows this board."""
    board = _board(db, payload.num_teams, payload.slots())
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["rank", "name", "position", "team", "season_points", "vorp", "tier", "adp", "source"])
    for bp in board[:400]:
        writer.writerow(
            [
                bp.overall_rank,
                bp.name,
                bp.position,
                bp.nfl_team,
                f"{bp.season_points:.1f}",
                bp.vorp,
                bp.tier,
                bp.adp or "",
                bp.source,
            ]
        )
    return buf.getvalue()


@router.get("/players")
def searchable_players(
    search: str = "",
    limit: int = 25,
    num_teams: int = 12,
    db: Session = Depends(get_db),
) -> dict:
    """Name search over the board, for the 'mark as drafted' box."""
    board = _board(db, num_teams, DEFAULT_ROSTER_POSITIONS)
    needle = assistant.normalize_name(search)
    matches = [bp for bp in board if not needle or needle in bp.norm_name]
    return {"players": [_entry(bp) for bp in matches[:limit]]}
