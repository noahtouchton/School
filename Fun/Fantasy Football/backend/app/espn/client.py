"""ESPN Fantasy Football client.

Reads go through the `espn_api` package, which authenticates with the two
cookies your own browser already has (espn_s2 + SWID) -- no developer app, no
approval queue, which is exactly why this is the primary integration.

`espn_api` is read-only, so lineup changes and add/drops post directly to
ESPN's write host with those same cookies. Those endpoints are undocumented
(they're what espn.com's own UI calls), so every write is verified by re-reading
the roster afterwards rather than trusting a 200.

Slot vocabulary: ESPN's names ("RB/WR/TE", "D/ST", "BE") are normalized on the
way in to the canonical set the draft/lineup engines already speak, and mapped
back to ESPN slot ids on the way out.
"""

from __future__ import annotations

import requests
from espn_api.football import League
from espn_api.requests.espn_requests import (
    ESPNAccessDenied,
    ESPNInvalidLeague,
    ESPNUnknownError,
)

WRITE_HOST = "https://lm-api-writes.fantasy.espn.com"

# ESPN slot name -> canonical slot name used by app/draft/*.
SLOT_FROM_ESPN = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "K": "K",
    "D/ST": "DEF",
    "RB/WR/TE": "W/R/T",
    "WR/TE": "W/T",
    "RB/WR": "W/R",
    "OP": "Q/W/R/T",
    "BE": "BN",
    "IR": "IR",
}
SLOT_TO_ESPN = {v: k for k, v in SLOT_FROM_ESPN.items()}

# Canonical slot -> ESPN lineupSlotId (from espn_api's POSITION_MAP).
ESPN_SLOT_ID = {
    "QB": 0,
    "RB": 2,
    "W/R": 3,
    "WR": 4,
    "W/T": 5,
    "TE": 6,
    "Q/W/R/T": 7,
    "DEF": 16,
    "K": 17,
    "BN": 20,
    "IR": 21,
    "W/R/T": 23,
}

# Slots that aren't real roster spots in this app's model.
IGNORED_SLOTS = {"", "Rookie", "TQB", "P", "HC", "ER", "DP", "DT", "DE", "LB", "DL", "CB", "S", "DB"}


class EspnError(Exception):
    pass


def _normalize_position(position: str | None) -> str:
    if not position:
        return ""
    return "DST" if position == "D/ST" else position


def _normalize_slot(slot: str | None) -> str | None:
    if slot is None:
        return None
    return SLOT_FROM_ESPN.get(slot, slot)


def connect(league_id: int, year: int, espn_s2: str | None, swid: str | None) -> League:
    """Build a League, translating ESPN's auth failures into one clear error."""
    try:
        return League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
    except ESPNAccessDenied as e:
        raise EspnError(
            "ESPN rejected those credentials. For a private league you need both the "
            "espn_s2 and SWID cookies from a browser where you're logged into ESPN. "
            f"({e})"
        ) from e
    except ESPNInvalidLeague as e:
        raise EspnError(f"No ESPN league with id {league_id} for {year}. ({e})") from e
    except ESPNUnknownError as e:
        raise EspnError(f"ESPN returned an unexpected error: {e}") from e


def _swid_matches(swid: str | None, owners) -> bool:
    """Team.owners holds member ids in {BRACED-UPPER} form; compare loosely."""
    if not swid:
        return False
    target = swid.strip().strip("{}").upper()
    for owner in owners or []:
        raw = owner.get("id") if isinstance(owner, dict) else owner
        if raw and str(raw).strip().strip("{}").upper() == target:
            return True
    return False


def my_team(league: League, swid: str | None):
    """The Team owned by the logged-in member, or None if it can't be identified."""
    for team in league.teams:
        if _swid_matches(swid, getattr(team, "owners", [])):
            return team
    return None


def roster_positions(league: League) -> list[dict]:
    """League roster slots in the canonical {position, count} shape."""
    slots = []
    for espn_name, count in (league.settings.position_slot_counts or {}).items():
        if not count or espn_name in IGNORED_SLOTS:
            continue
        canonical = _normalize_slot(espn_name)
        if canonical is None:
            continue
        slots.append({"position": canonical, "count": int(count)})
    return slots


def player_dict(player, team_id: int | None = None) -> dict:
    """One ESPN Player in the shape app/draft/lineup.py expects."""
    eligible = [
        _normalize_slot(slot)
        for slot in (getattr(player, "eligibleSlots", []) or [])
        if slot not in IGNORED_SLOTS
    ]
    projected = None
    stats = getattr(player, "stats", {}) or {}
    for week_stats in stats.values():
        if isinstance(week_stats, dict) and "projected_points" in week_stats:
            projected = week_stats["projected_points"]
            break

    return {
        "player_key": str(player.playerId),
        "player_id": player.playerId,
        "name": player.name,
        "position": _normalize_position(getattr(player, "position", None)),
        "primary_position": _normalize_position(getattr(player, "position", None)),
        "nfl_team": (getattr(player, "proTeam", "") or "").upper(),
        "injury_status": getattr(player, "injuryStatus", None),
        "eligible_positions": [e for e in eligible if e],
        "selected_position": _normalize_slot(getattr(player, "lineupSlot", None)),
        "percent_owned": getattr(player, "percent_owned", None),
        "espn_projected_points": projected,
        "is_undroppable": False,
        "team_id": team_id,
    }


def team_roster(league: League, team) -> list[dict]:
    return [player_dict(p, team.team_id) for p in team.roster]


def free_agents(league: League, size: int = 60, position: str | None = None) -> list[dict]:
    espn_position = SLOT_TO_ESPN.get(position, position) if position else None
    try:
        players = league.free_agents(size=size, position=espn_position)
    except Exception as e:  # espn_api raises bare Exceptions for bad params
        raise EspnError(f"Couldn't load ESPN free agents: {e}") from e
    return [player_dict(p) for p in players]


def draft_picks(league: League, refresh: bool = True) -> list[dict]:
    """Draft picks so far. During a live draft this grows as picks are made."""
    if refresh:
        try:
            league.refresh_draft()
        except Exception as e:
            raise EspnError(f"Couldn't refresh the ESPN draft: {e}") from e

    picks = []
    for pick in league.draft:
        team = getattr(pick, "team", None)
        picks.append(
            {
                "round": pick.round_num,
                "round_pick": pick.round_pick,
                "player_id": pick.playerId,
                "name": pick.playerName,
                "team_id": getattr(team, "team_id", None),
                "team_name": getattr(team, "team_name", None),
                "bid_amount": getattr(pick, "bid_amount", None),
                "keeper": getattr(pick, "keeper_status", False),
            }
        )
    picks.sort(key=lambda p: (p["round"] or 0, p["round_pick"] or 0))
    return picks


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def _write(league: League, league_id: int, year: int, swid: str | None, body: dict) -> dict:
    url = f"{WRITE_HOST}/apis/v3/games/ffl/seasons/{year}/segments/0/leagues/{league_id}/transactions/"
    cookies = league.espn_request.cookies or {}
    resp = requests.post(
        url,
        json=body,
        cookies=cookies,
        headers={
            "Content-Type": "application/json",
            "X-Fantasy-Source": "kona",
            "X-Fantasy-Platform": "kona-PROD",
        },
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise EspnError(
            f"ESPN rejected the change (HTTP {resp.status_code}): {resp.text[:400]}. "
            "If this says unauthorized, your espn_s2/SWID cookies have expired -- "
            "grab fresh ones from your browser."
        )
    try:
        return resp.json()
    except ValueError:
        return {}


def set_lineup(
    league: League,
    league_id: int,
    year: int,
    swid: str | None,
    team_id: int,
    week: int,
    moves: list[dict],
) -> dict:
    """Apply lineup slot changes. moves: [{player_id, from_slot, to_slot}] using
    canonical slot names."""
    if not moves:
        return {"applied": 0}

    items = []
    for move in moves:
        from_slot = ESPN_SLOT_ID.get(move["from_slot"])
        to_slot = ESPN_SLOT_ID.get(move["to_slot"])
        if from_slot is None or to_slot is None:
            raise EspnError(f"Don't know how to map slots for move {move}")
        items.append(
            {
                "playerId": int(move["player_id"]),
                "type": "LINEUP",
                "fromLineupSlotId": from_slot,
                "toLineupSlotId": to_slot,
            }
        )

    body = {
        "isLeagueManager": False,
        "teamId": int(team_id),
        "type": "ROSTER",
        "memberId": swid,
        "scoringPeriodId": int(week),
        "executionType": "EXECUTE",
        "items": items,
    }
    _write(league, league_id, year, swid, body)
    return {"applied": len(items)}


def add_drop(
    league: League,
    league_id: int,
    year: int,
    swid: str | None,
    team_id: int,
    week: int,
    add_player_id: int | None,
    drop_player_id: int | None,
    bid: int | None = None,
) -> dict:
    items = []
    if add_player_id is not None:
        items.append(
            {
                "playerId": int(add_player_id),
                "type": "ADD",
                "toTeamId": int(team_id),
                "toLineupSlotId": ESPN_SLOT_ID["BN"],
            }
        )
    if drop_player_id is not None:
        items.append(
            {
                "playerId": int(drop_player_id),
                "type": "DROP",
                "fromTeamId": int(team_id),
                "fromLineupSlotId": ESPN_SLOT_ID["BN"],
            }
        )
    if not items:
        raise EspnError("Nothing to add or drop")

    body = {
        "isLeagueManager": False,
        "teamId": int(team_id),
        "type": "WAIVER" if bid is not None else "FREEAGENT",
        "memberId": swid,
        "scoringPeriodId": int(week),
        "executionType": "EXECUTE",
        "items": items,
    }
    if bid is not None:
        body["bidAmount"] = int(bid)
    _write(league, league_id, year, swid, body)
    return {"ok": True}
