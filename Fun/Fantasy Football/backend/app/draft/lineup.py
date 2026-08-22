"""Weekly lineup optimization + waiver-wire scanning for a linked Yahoo team.

Both features hinge on one lookup: normalized (name, position) -> our model's
per-game projection for the latest generated week. Yahoo tells us who's on the
roster / the wire and which slots they're eligible for; our model says who
scores. K/DEF have no model projections and are handled conservatively (keep
whoever is already slotted rather than churn on zero information).
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Player, PlayerWeeklyProjection
from app.draft.assistant import normalize_name
from app.ml.train import MODEL_VERSION

# Yahoo slot names ordered most-restrictive first so the greedy fill never
# burns a flex-eligible stud on a dedicated slot another player could take.
SLOT_ORDER = ["QB", "K", "DEF", "TE", "RB", "WR", "W/T", "W/R", "W/R/T", "Q/W/R/T"]


def projection_lookup(db: Session) -> tuple[dict[tuple[str, str], float], int | None, int | None]:
    """(norm_name, position) -> projected points for the latest projection week."""
    season = db.execute(
        select(func.max(PlayerWeeklyProjection.season)).where(
            PlayerWeeklyProjection.model_version == MODEL_VERSION
        )
    ).scalar_one_or_none()
    if season is None:
        return {}, None, None
    week = db.execute(
        select(func.max(PlayerWeeklyProjection.week)).where(
            PlayerWeeklyProjection.model_version == MODEL_VERSION,
            PlayerWeeklyProjection.season == season,
        )
    ).scalar_one_or_none()

    rows = db.execute(
        select(Player.name, Player.position, PlayerWeeklyProjection.projected_points).join(
            PlayerWeeklyProjection,
            (PlayerWeeklyProjection.player_id == Player.id)
            & (PlayerWeeklyProjection.season == season)
            & (PlayerWeeklyProjection.week == week)
            & (PlayerWeeklyProjection.model_version == MODEL_VERSION),
        )
    ).all()
    lookup = {(normalize_name(name), pos): float(pts) for name, pos, pts in rows}
    return lookup, season, week


def _proj_for(player: dict, lookup: dict[tuple[str, str], float]) -> float | None:
    pos = player.get("primary_position") or player.get("position") or ""
    if pos == "DEF":
        return None
    return lookup.get((normalize_name(player.get("name") or ""), pos))


def optimal_lineup(
    roster: list[dict],
    roster_positions: list[dict],
    lookup: dict[tuple[str, str], float],
) -> dict:
    """Greedy best lineup for a Yahoo roster. Returns starters, bench, and the
    position changes needed to get there from the current lineup."""
    slots: list[str] = []
    for rp in roster_positions:
        pos, count = rp.get("position"), rp.get("count", 0)
        if pos in ("BN", "IR", "IR+") or not pos:
            continue
        slots.extend([pos] * count)
    slots.sort(key=lambda s: SLOT_ORDER.index(s) if s in SLOT_ORDER else 99)

    # Players currently on IR stay put — moving them is a separate decision.
    active = [p for p in roster if p.get("selected_position") not in ("IR", "IR+")]
    projections = {p["player_key"]: _proj_for(p, lookup) for p in active}

    assigned: dict[str, str] = {}  # player_key -> slot
    for slot in slots:
        candidates = [
            p
            for p in active
            if p["player_key"] not in assigned and slot in (p.get("eligible_positions") or [])
        ]
        if not candidates:
            continue

        def sort_key(p: dict):
            proj = projections.get(p["player_key"])
            currently_here = p.get("selected_position") == slot
            # Highest projection wins; unknown projections (K/DEF) rank below
            # any projected player but the incumbent beats a challenger.
            return (proj if proj is not None else -1.0, currently_here)

        best = max(candidates, key=sort_key)
        assigned[best["player_key"]] = slot

    starters, bench, changes = [], [], []
    projected_total = 0.0
    for p in active:
        slot = assigned.get(p["player_key"], "BN")
        proj = projections.get(p["player_key"])
        entry = {
            "player_key": p["player_key"],
            "name": p.get("name"),
            "position": p.get("primary_position") or p.get("position"),
            "nfl_team": p.get("nfl_team"),
            "slot": slot,
            "current_slot": p.get("selected_position"),
            "projected_points": proj,
            "injury_status": p.get("injury_status"),
            "bye_week": p.get("bye_week"),
        }
        if slot == "BN":
            bench.append(entry)
        else:
            starters.append(entry)
            projected_total += proj or 0.0
        if slot != p.get("selected_position"):
            changes.append({"player_key": p["player_key"], "name": p.get("name"), "from": p.get("selected_position"), "to": slot})

    current_total = sum(
        (projections.get(p["player_key"]) or 0.0)
        for p in active
        if p.get("selected_position") not in ("BN", "IR", "IR+")
    )
    starters.sort(key=lambda e: (SLOT_ORDER.index(e["slot"]) if e["slot"] in SLOT_ORDER else 99))
    bench.sort(key=lambda e: -(e["projected_points"] or 0.0))

    return {
        "starters": starters,
        "bench": bench,
        "changes": changes,
        "projected_total": round(projected_total, 1),
        "current_total": round(current_total, 1),
        "improvement": round(projected_total - current_total, 1),
    }


def lineup_change_payload(result: dict) -> list[dict]:
    """Full slot assignment payload for the Yahoo roster PUT (Yahoo wants every
    moved player's new position; sending all active players is safest)."""
    payload = [
        {"player_key": e["player_key"], "position": e["slot"]}
        for e in result["starters"] + result["bench"]
    ]
    return payload


def waiver_recommendations(
    free_agents: list[dict],
    my_roster: list[dict],
    lookup: dict[tuple[str, str], float],
    min_gain: float = 1.0,
) -> list[dict]:
    """Free agents projected to outscore my weakest same-position player."""
    recs = []
    my_by_pos: dict[str, list[dict]] = {}
    for p in my_roster:
        pos = p.get("primary_position") or p.get("position") or ""
        my_by_pos.setdefault(pos, []).append(p)

    for fa in free_agents:
        pos = fa.get("primary_position") or fa.get("position") or ""
        fa_proj = _proj_for(fa, lookup)
        if fa_proj is None:
            continue
        mine = my_by_pos.get(pos, [])
        if not mine:
            continue
        # Weakest of my players at that position (unknown projection = 0-ish).
        weakest = min(mine, key=lambda p: _proj_for(p, lookup) or 0.0)
        weakest_proj = _proj_for(weakest, lookup) or 0.0
        gain = fa_proj - weakest_proj
        if gain < min_gain:
            continue
        recs.append(
            {
                "add": {
                    "player_key": fa.get("player_key"),
                    "name": fa.get("name"),
                    "position": pos,
                    "nfl_team": fa.get("nfl_team"),
                    "projected_points": round(fa_proj, 1),
                    "percent_owned": fa.get("percent_owned"),
                    "ownership_type": fa.get("ownership_type"),
                    "injury_status": fa.get("injury_status"),
                },
                "drop": {
                    "player_key": weakest.get("player_key"),
                    "name": weakest.get("name"),
                    "position": pos,
                    "projected_points": round(weakest_proj, 1),
                    "is_undroppable": weakest.get("is_undroppable", False),
                },
                "projected_gain": round(gain, 1),
                "suggested_faab": min(30, max(1, round(gain * 3))),
            }
        )
    recs.sort(key=lambda r: -r["projected_gain"])
    return recs
