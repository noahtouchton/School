"""Thin client for the Yahoo Fantasy Sports API (v2).

Reads use `?format=json`. Yahoo's JSON is a direct translation of their XML —
resources are lists whose first element is the metadata (either a dict or a
list of single-key dicts) and whose later elements are named sub-resources;
collections are dicts keyed by "0", "1", ... plus a "count". The helpers at the
top of this module normalize that so the rest of the codebase never sees it.

Writes (transactions, lineup changes) must be XML — Yahoo does not accept JSON
bodies for mutations.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import requests

from app.yahoo.oauth import get_access_token

BASE = "https://fantasysports.yahooapis.com/fantasy/v2"


class YahooApiError(Exception):
    pass


# ---------------------------------------------------------------------------
# JSON normalization helpers
# ---------------------------------------------------------------------------

def _merge(attr_list) -> dict:
    """Yahoo represents a resource's attributes as a list of single-key dicts,
    with stray empty strings/lists mixed in. Flatten into one dict."""
    out: dict = {}
    if isinstance(attr_list, dict):
        return attr_list
    for item in attr_list or []:
        if isinstance(item, dict):
            out.update(item)
        elif isinstance(item, list):
            out.update(_merge(item))
    return out


def _items(collection, key: str) -> list:
    """{"0": {"player": ...}, "1": ..., "count": n} -> [value_of_key, ...]"""
    if not isinstance(collection, dict):
        return []
    out = []
    for idx in sorted((k for k in collection if k.isdigit()), key=int):
        node = collection[idx]
        if isinstance(node, dict) and key in node:
            out.append(node[key])
    return out


def _sub(resource_list, name: str):
    """Named sub-resource from a resource list's tail elements."""
    if isinstance(resource_list, dict):
        return resource_list.get(name)
    for el in resource_list[1:] if isinstance(resource_list, list) else []:
        if isinstance(el, dict) and name in el:
            return el[name]
    return None


def _meta(resource_list) -> dict:
    if isinstance(resource_list, dict):
        return resource_list
    if isinstance(resource_list, list) and resource_list:
        return _merge(resource_list[0]) if isinstance(resource_list[0], list) else resource_list[0]
    return {}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(path: str) -> dict:
    token = get_access_token()
    sep = "&" if "?" in path else "?"
    resp = requests.get(
        f"{BASE}/{path}{sep}format=json",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise YahooApiError(f"Yahoo API GET /{path} failed ({resp.status_code}): {resp.text[:400]}")
    return resp.json().get("fantasy_content", {})


def _send_xml(method: str, path: str, xml_body: str) -> str:
    token = get_access_token()
    resp = requests.request(
        method,
        f"{BASE}/{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/xml"},
        data=xml_body.encode("utf-8"),
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise YahooApiError(f"Yahoo API {method} /{path} failed ({resp.status_code}): {resp.text[:600]}")
    return resp.text


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_player(player_node) -> dict:
    """A player resource: [ [attr dicts...], {"selected_position": [...]}, ... ]"""
    attrs = _merge(player_node[0] if isinstance(player_node, list) else player_node)
    name = attrs.get("name") or {}
    bye = attrs.get("bye_weeks") or {}
    eligible = [
        d.get("position")
        for d in (attrs.get("eligible_positions") or [])
        if isinstance(d, dict) and d.get("position")
    ]
    player = {
        "player_key": attrs.get("player_key"),
        "player_id": attrs.get("player_id"),
        "name": name.get("full"),
        "position": attrs.get("display_position"),
        "primary_position": attrs.get("primary_position"),
        "nfl_team": (attrs.get("editorial_team_abbr") or "").upper(),
        "injury_status": attrs.get("status"),  # Q / D / O / IR / ... or absent
        "bye_week": bye.get("week"),
        "eligible_positions": eligible,
        "is_undroppable": attrs.get("is_undroppable") == "1",
    }
    sel = _sub(player_node, "selected_position")
    if sel is not None:
        player["selected_position"] = _merge(sel).get("position")
    owned = _sub(player_node, "percent_owned")
    if owned is not None:
        player["percent_owned"] = _merge(owned).get("value")
    ownership = _sub(player_node, "ownership")
    if ownership is not None:
        own = _merge(ownership)
        player["ownership_type"] = own.get("ownership_type")
        player["waiver_date"] = own.get("waiver_date")
    return player


def _parse_team(team_node) -> dict:
    attrs = _meta(team_node)
    managers = [
        _merge(m.get("manager", {}))
        for m in (attrs.get("managers") or [])
        if isinstance(m, dict)
    ]
    return {
        "team_key": attrs.get("team_key"),
        "team_id": attrs.get("team_id"),
        "name": attrs.get("name"),
        "is_mine": str(attrs.get("is_owned_by_current_login", "0")) == "1",
        "draft_position": attrs.get("draft_position"),
        "faab_balance": attrs.get("faab_balance"),
        "waiver_priority": attrs.get("waiver_priority"),
        "number_of_moves": attrs.get("number_of_moves"),
        "managers": [m.get("nickname") for m in managers if m.get("nickname")],
        "logo": next(
            (
                tl.get("team_logo", {}).get("url")
                for tl in (attrs.get("team_logos") or [])
                if isinstance(tl, dict)
            ),
            None,
        ),
    }


def _parse_league_meta(meta: dict) -> dict:
    return {
        "league_key": meta.get("league_key"),
        "league_id": meta.get("league_id"),
        "name": meta.get("name"),
        "season": int(meta.get("season", 0) or 0),
        "num_teams": int(meta.get("num_teams", 0) or 0),
        "scoring_type": meta.get("scoring_type"),
        "draft_status": meta.get("draft_status"),  # predraft | draftactive | postdraft
        "current_week": meta.get("current_week"),
        "start_week": meta.get("start_week"),
        "end_week": meta.get("end_week"),
        "url": meta.get("url"),
        "is_finished": str(meta.get("is_finished", "0")) == "1",
    }


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_user_leagues() -> list[dict]:
    """All NFL fantasy leagues for the logged-in Yahoo user, newest season first."""
    content = _get("users;use_login=1/games;game_codes=nfl/leagues")
    leagues: list[dict] = []
    for user in _items(content.get("users", {}), "user"):
        games = _sub(user, "games")
        for game in _items(games or {}, "game"):
            league_coll = _sub(game, "leagues")
            for league in _items(league_coll or {}, "league"):
                leagues.append(_parse_league_meta(_meta(league)))
    leagues.sort(key=lambda l: l["season"], reverse=True)
    return leagues


def get_league_settings(league_key: str) -> dict:
    content = _get(f"league/{league_key}/settings")
    league = content.get("league", [])
    out = _parse_league_meta(_meta(league))

    settings = _sub(league, "settings")
    settings_attrs = _merge(settings) if settings else {}

    roster_positions = []
    for rp in settings_attrs.get("roster_positions") or []:
        if isinstance(rp, dict) and "roster_position" in rp:
            pos = rp["roster_position"]
            roster_positions.append(
                {
                    "position": pos.get("position"),
                    "position_type": pos.get("position_type"),
                    "count": int(pos.get("count", 0) or 0),
                }
            )
    out["roster_positions"] = roster_positions
    out["uses_faab"] = str(settings_attrs.get("uses_faab", "0")) == "1"
    out["draft_time"] = settings_attrs.get("draft_time")
    out["draft_type"] = settings_attrs.get("draft_type")

    # Stat modifiers -> a {stat_id: value} map plus a best-effort PPR detection.
    modifiers = {}
    stats_node = (settings_attrs.get("stat_modifiers") or {}).get("stats") or []
    for s in stats_node:
        if isinstance(s, dict) and "stat" in s:
            stat = s["stat"]
            try:
                modifiers[int(stat.get("stat_id"))] = float(stat.get("value", 0))
            except (TypeError, ValueError):
                continue
    out["stat_modifiers"] = modifiers
    # Yahoo stat id 11 = receptions.
    out["ppr_value"] = modifiers.get(11, 0.0)
    return out


def get_league_teams(league_key: str) -> list[dict]:
    content = _get(f"league/{league_key}/teams")
    league = content.get("league", [])
    teams_coll = _sub(league, "teams")
    return [_parse_team(t) for t in _items(teams_coll or {}, "team")]


def get_my_team_key(league_key: str) -> str | None:
    for team in get_league_teams(league_key):
        if team["is_mine"]:
            return team["team_key"]
    return None


def get_team_roster(team_key: str, week: int | None = None) -> list[dict]:
    path = f"team/{team_key}/roster"
    if week is not None:
        path += f";week={week}"
    content = _get(path)
    team = content.get("team", [])
    roster = _sub(team, "roster")
    if roster is None:
        return []
    players_coll = _sub(roster, "players") if isinstance(roster, list) else roster.get("0", {}).get("players")
    if players_coll is None and isinstance(roster, dict):
        players_coll = roster.get("players")
    return [_parse_player(p) for p in _items(players_coll or {}, "player")]


def get_draft_results(league_key: str) -> list[dict]:
    """Draft picks so far. During a live draft this updates as picks are made."""
    content = _get(f"league/{league_key}/draftresults")
    league = content.get("league", [])
    results_coll = _sub(league, "draft_results")
    picks = []
    for result in _items(results_coll or {}, "draft_result"):
        attrs = _meta(result)
        if not attrs.get("player_key"):
            continue  # unfilled future pick slots
        picks.append(
            {
                "pick": int(attrs.get("pick", 0) or 0),
                "round": int(attrs.get("round", 0) or 0),
                "team_key": attrs.get("team_key"),
                "player_key": attrs.get("player_key"),
            }
        )
    picks.sort(key=lambda p: p["pick"])
    return picks


def get_players_by_keys(player_keys: list[str]) -> list[dict]:
    """Resolve player keys to names/positions. Batched (Yahoo caps ~25 per call)."""
    out: list[dict] = []
    for i in range(0, len(player_keys), 25):
        batch = player_keys[i : i + 25]
        if not batch:
            continue
        content = _get(f"players;player_keys={','.join(batch)}")
        players_coll = content.get("players", {})
        out.extend(_parse_player(p) for p in _items(players_coll, "player"))
    return out


def get_league_players(
    league_key: str,
    status: str = "FA",
    position: str | None = None,
    count: int = 50,
    sort: str = "AR",
) -> list[dict]:
    """Players in a league context (status: FA free agents, W waivers, T taken, A all)."""
    out: list[dict] = []
    start = 0
    while len(out) < count:
        page = min(25, count - len(out))
        path = f"league/{league_key}/players;status={status};sort={sort};start={start};count={page}"
        if position:
            path += f";position={position}"
        content = _get(path)
        league = content.get("league", [])
        players_coll = _sub(league, "players")
        players = [_parse_player(p) for p in _items(players_coll or {}, "player")]
        if not players:
            break
        out.extend(players)
        start += len(players)
        if len(players) < page:
            break
    return out


# ---------------------------------------------------------------------------
# Writes (XML)
# ---------------------------------------------------------------------------

def _transaction_xml(
    add_player_key: str | None,
    drop_player_key: str | None,
    team_key: str,
    faab_bid: int | None,
) -> str:
    if add_player_key and drop_player_key:
        ttype = "add/drop"
        players = f"""
    <players>
      <player>
        <player_key>{escape(add_player_key)}</player_key>
        <transaction_data>
          <type>add</type>
          <destination_team_key>{escape(team_key)}</destination_team_key>
        </transaction_data>
      </player>
      <player>
        <player_key>{escape(drop_player_key)}</player_key>
        <transaction_data>
          <type>drop</type>
          <source_team_key>{escape(team_key)}</source_team_key>
        </transaction_data>
      </player>
    </players>"""
    elif add_player_key:
        ttype = "add"
        players = f"""
    <player>
      <player_key>{escape(add_player_key)}</player_key>
      <transaction_data>
        <type>add</type>
        <destination_team_key>{escape(team_key)}</destination_team_key>
      </transaction_data>
    </player>"""
    elif drop_player_key:
        ttype = "drop"
        players = f"""
    <player>
      <player_key>{escape(drop_player_key)}</player_key>
      <transaction_data>
        <type>drop</type>
        <source_team_key>{escape(team_key)}</source_team_key>
      </transaction_data>
    </player>"""
    else:
        raise YahooApiError("Transaction needs at least one of add/drop player keys")

    faab = f"\n    <faab_bid>{int(faab_bid)}</faab_bid>" if faab_bid is not None else ""
    return f"""<?xml version="1.0"?>
<fantasy_content>
  <transaction>
    <type>{ttype}</type>{faab}{players}
  </transaction>
</fantasy_content>"""


def execute_transaction(
    league_key: str,
    team_key: str,
    add_player_key: str | None = None,
    drop_player_key: str | None = None,
    faab_bid: int | None = None,
) -> str:
    """Add and/or drop a player (waiver claims included — Yahoo routes them
    automatically based on the player's waiver state)."""
    xml = _transaction_xml(add_player_key, drop_player_key, team_key, faab_bid)
    return _send_xml("POST", f"league/{league_key}/transactions", xml)


def set_lineup(team_key: str, week: int, positions: list[dict]) -> str:
    """Set the weekly lineup. positions: [{"player_key": ..., "position": "WR"|"BN"|...}]"""
    players_xml = "\n".join(
        f"""      <player>
        <player_key>{escape(p["player_key"])}</player_key>
        <position>{escape(p["position"])}</position>
      </player>"""
        for p in positions
    )
    xml = f"""<?xml version="1.0"?>
<fantasy_content>
  <roster>
    <coverage_type>week</coverage_type>
    <week>{int(week)}</week>
    <players>
{players_xml}
    </players>
  </roster>
</fantasy_content>"""
    return _send_xml("PUT", f"team/{team_key}/roster", xml)
