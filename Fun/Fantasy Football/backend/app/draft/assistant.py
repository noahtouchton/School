"""Value-based draft engine for a real (Yahoo) league.

Builds a season-long draft board from our own historical data, uses consensus
ADP to (a) value players we have no history for (rookies) and (b) pull outliers
back toward sanity, then converts raw points into VORP (value over the
replacement-level player at that position given this league's size and roster
slots). During a live draft it re-scores the remaining board against the picks
already made and my roster's actual needs.

Why this does NOT reuse the weekly ML projections
-------------------------------------------------
app/ml/predict.py answers "what will this player score THIS week", and its
dominant feature is the player's rolling 8-game average. That's the right
question in-season (lineup.py uses exactly those numbers), but extrapolating
one such projection across a full season is actively misleading at draft time:
a backup who took over for an injured starter in December carries eight
inflated games into the average, and x16 turns that into a first-round
ranking. Season-long value instead aggregates every game from the last two
seasons, weights the recent one higher, and regresses thin samples toward the
position's replacement level so a hot streak can't masquerade as a season.

Everything the model can't know (K/DST value, rookies) leans on the market.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.scoring import calculate_fantasy_points
from app.db.models import ConsensusAdp, Player, PlayerWeeklyStat

# A fantasy season is 17 weeks with one bye, so a drafted starter plays ~16.
GAMES_PER_SEASON = 16

# When both a history-based value and an ADP-implied value exist, trust our own
# data more but let the market pull hard outliers back toward sanity.
MODEL_WEIGHT = 0.7

# Seasons of history that feed draft value, and how much the older one counts.
HISTORY_SEASONS = 2
PRIOR_SEASON_WEIGHT = 0.45

# Games of "prior belief" blended into every player's per-game average. A
# player with three huge games ends up mostly prior (i.e. near replacement
# level), which is the entire point -- three games is not evidence of a season.
SHRINKAGE_GAMES = 6.0

# Percentile of qualified per-game scoring used as each position's shrinkage
# target: roughly a startable-but-unremarkable player at that position.
REPLACEMENT_PERCENTILE = 0.40
QUALIFIED_GAMES = 8.0

# How flex slots split across the positions eligible for them, roughly matching
# how leagues actually fill W/R/T slots.
FLEX_SHARES = {
    "W/R/T": {"RB": 0.35, "WR": 0.5, "TE": 0.15},
    "W/T": {"WR": 0.75, "TE": 0.25},
    "W/R": {"RB": 0.45, "WR": 0.55},
    "Q/W/R/T": {"QB": 0.8, "RB": 0.08, "WR": 0.1, "TE": 0.02},  # superflex ≈ QB2
}

# Season-point baselines for positions our models don't cover. Only their
# relative late-round ordering matters; ADP rank sets the decline.
NO_MODEL_BASELINES = {"K": 128.0, "DST": 120.0}
NO_MODEL_DECLINE_PER_RANK = 2.5

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

_SUFFIX_RE = re.compile(r"\s+(jr\.?|sr\.?|i{2,3}|iv|v)$", re.IGNORECASE)


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes so 'A.J. Brown Jr.' == 'aj brown'."""
    name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    name = _SUFFIX_RE.sub("", name.strip())
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


@dataclass
class BoardPlayer:
    name: str
    position: str
    nfl_team: str | None
    season_points: float
    source: str  # "model" | "blend" | "market" | "baseline"
    adp: float | None = None
    per_game: float | None = None
    injury_status: str | None = None
    player_id: str | None = None
    vorp: float = 0.0
    tier: int = 1
    overall_rank: int = 0
    position_rank: int = 0
    age: int | None = None
    # Filled in by the season simulator, which knows the schedule; None outside it.
    bye_week: int | None = None
    norm_name: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.norm_name:
            self.norm_name = normalize_name(self.name)

    @property
    def key(self) -> str:
        """Stable identity. player_id is absent for ADP-only entries (rookies with
        no stat history), so fall back to the normalized name + position."""
        return self.player_id or f"{self.norm_name}|{self.position}"


def _shrunk_season_ppg(
    db: Session, through_season: int | None = None
) -> dict[str, tuple[Player, float, float]]:
    """player_id -> (player, shrunk points-per-game, weighted games of evidence).

    Weighted games double as a confidence measure: a player with a handful of
    games sits close to his position's replacement level no matter how well
    those games went.

    through_season caps the history used. Simulated seasons pass the season
    before the one being replayed, so a backtest can't draft on knowledge of how
    that season actually turned out.
    """
    stmt = select(func.max(PlayerWeeklyStat.season))
    if through_season is not None:
        stmt = stmt.where(PlayerWeeklyStat.season <= through_season)
    latest_season = db.execute(stmt).scalar_one_or_none()
    if latest_season is None:
        return {}
    seasons = [latest_season - offset for offset in range(HISTORY_SEASONS)]

    players = {
        p.id: p
        for p in db.execute(select(Player).where(Player.position.in_(SKILL_POSITIONS)))
        .scalars()
        .all()
    }

    totals: dict[str, list[float]] = {}  # player_id -> [weighted points, weighted games]
    rows = (
        db.execute(select(PlayerWeeklyStat).where(PlayerWeeklyStat.season.in_(seasons)))
        .scalars()
        .all()
    )
    for row in rows:
        if row.player_id not in players:
            continue
        weight = 1.0 if row.season == latest_season else PRIOR_SEASON_WEIGHT
        points = calculate_fantasy_points(row.stats)
        bucket = totals.setdefault(row.player_id, [0.0, 0.0])
        bucket[0] += points * weight
        bucket[1] += weight

    # Replacement level per position, from players with enough games to judge.
    qualified: dict[str, list[float]] = {}
    for player_id, (pts, games) in totals.items():
        if games >= QUALIFIED_GAMES:
            qualified.setdefault(players[player_id].position, []).append(pts / games)
    replacement: dict[str, float] = {}
    for position, values in qualified.items():
        values.sort()
        idx = min(len(values) - 1, int(len(values) * REPLACEMENT_PERCENTILE))
        replacement[position] = values[idx]
    fallback = statistics.median(replacement.values()) if replacement else 5.0

    out: dict[str, tuple[Player, float, float]] = {}
    for player_id, (pts, games) in totals.items():
        if games <= 0:
            continue
        player = players[player_id]
        prior = replacement.get(player.position, fallback)
        shrunk = (pts + SHRINKAGE_GAMES * prior) / (games + SHRINKAGE_GAMES)
        out[player_id] = (player, shrunk, games)
    return out


def _adp_rows(db: Session, through_season: int | None = None) -> list[ConsensusAdp]:
    stmt = select(func.max(ConsensusAdp.season))
    if through_season is not None:
        stmt = stmt.where(ConsensusAdp.season <= through_season)
    season = db.execute(stmt).scalar_one_or_none()
    if season is None:
        return []
    return list(
        db.execute(select(ConsensusAdp).where(ConsensusAdp.season == season)).scalars().all()
    )


def _adp_implied_points(
    with_both: list[tuple[float, float]], adp: float
) -> float | None:
    """Given (adp, model_points) pairs for a position, estimate points for a
    player who only has an ADP. Piecewise-linear on the ADP-sorted curve,
    forced monotone non-increasing so a better ADP never implies fewer points."""
    if len(with_both) < 4:
        return None
    pairs = sorted(with_both)
    # Enforce monotonicity: running max from the right.
    points = [p for _, p in pairs]
    for i in range(len(points) - 2, -1, -1):
        points[i] = max(points[i], points[i + 1])
    adps = [a for a, _ in pairs]

    if adp <= adps[0]:
        return points[0]
    if adp >= adps[-1]:
        # Decay beyond the fitted range instead of going flat.
        return max(points[-1] - (adp - adps[-1]) * 0.35, 0.0)
    for i in range(1, len(adps)):
        if adp <= adps[i]:
            span = adps[i] - adps[i - 1] or 1.0
            frac = (adp - adps[i - 1]) / span
            return points[i - 1] + frac * (points[i] - points[i - 1])
    return points[-1]


def build_board(db: Session, through_season: int | None = None) -> list[BoardPlayer]:
    """Season-long draft board across all fantasy positions.

    through_season restricts the history (and ADP) used, so a simulated replay of
    a past season drafts with only what was knowable beforehand.
    """
    history = _shrunk_season_ppg(db, through_season)

    adp_by_player_id: dict[str, float] = {}
    adp_only: list[ConsensusAdp] = []
    for row in _adp_rows(db, through_season):
        if row.player_id and row.player_id in history:
            adp_by_player_id[row.player_id] = row.adp
        else:
            adp_only.append(row)

    board: list[BoardPlayer] = []
    curve_data: dict[str, list[tuple[float, float]]] = {}

    for player_id, (player, per_game, games) in history.items():
        season_pts = per_game * GAMES_PER_SEASON
        adp = adp_by_player_id.get(player_id)
        # Only well-established players anchor the ADP->points curve; fitting it
        # through thin-sample players would bake their shrinkage back in.
        if adp is not None and games >= QUALIFIED_GAMES:
            curve_data.setdefault(player.position, []).append((adp, season_pts))
        board.append(
            BoardPlayer(
                name=player.name,
                position=player.position,
                nfl_team=player.nfl_team,
                season_points=season_pts,
                per_game=per_game,
                source="model",
                adp=adp,
                injury_status=player.injury_status,
                player_id=player_id,
                age=player.age,
            )
        )

    # Blend market into modeled players that have an ADP.
    for bp in board:
        if bp.adp is not None:
            implied = _adp_implied_points(curve_data.get(bp.position, []), bp.adp)
            if implied is not None:
                bp.season_points = MODEL_WEIGHT * bp.season_points + (1 - MODEL_WEIGHT) * implied
                bp.source = "blend"

    # ADP-only players: rookies/no-history skill players get curve-implied
    # value; K/DST get baseline values ordered by ADP rank.
    no_model_rank: dict[str, int] = {}
    seen = {(bp.norm_name, bp.position) for bp in board}
    for row in sorted(adp_only, key=lambda r: r.adp):
        norm = normalize_name(row.name_raw)
        if (norm, row.position) in seen:
            continue
        if row.position in SKILL_POSITIONS:
            implied = _adp_implied_points(curve_data.get(row.position, []), row.adp)
            if implied is None:
                continue
            board.append(
                BoardPlayer(
                    name=row.name_raw,
                    position=row.position,
                    nfl_team=row.nfl_team,
                    season_points=implied,
                    per_game=implied / GAMES_PER_SEASON,
                    source="market",
                    adp=row.adp,
                    player_id=row.player_id,
                )
            )
        elif row.position in NO_MODEL_BASELINES:
            rank = no_model_rank.get(row.position, 0)
            no_model_rank[row.position] = rank + 1
            board.append(
                BoardPlayer(
                    name=row.name_raw,
                    position=row.position,
                    nfl_team=row.nfl_team,
                    season_points=NO_MODEL_BASELINES[row.position] - rank * NO_MODEL_DECLINE_PER_RANK,
                    source="baseline",
                    adp=row.adp,
                    player_id=row.player_id,
                )
            )
        seen.add((norm, row.position))

    return board


# ---------------------------------------------------------------------------
# League-aware valuation
# ---------------------------------------------------------------------------

def starters_by_position(roster_positions: list[dict]) -> dict[str, float]:
    """Effective starting slots per position, with flex slots split fractionally."""
    starters: dict[str, float] = {}
    for slot in roster_positions:
        pos, count = slot.get("position"), slot.get("count", 0)
        if not pos or pos in ("BN", "IR", "IR+"):
            continue
        if pos in FLEX_SHARES:
            for p, share in FLEX_SHARES[pos].items():
                starters[p] = starters.get(p, 0.0) + share * count
        else:
            key = "DST" if pos == "DEF" else pos
            starters[key] = starters.get(key, 0.0) + count
    return starters


def apply_vorp(board: list[BoardPlayer], roster_positions: list[dict], num_teams: int) -> None:
    """Set vorp/tier/ranks in place. Replacement level = the player who'd be the
    best one left on waivers once every team fills its starters plus a bit of bench."""
    starters = starters_by_position(roster_positions)
    by_pos: dict[str, list[BoardPlayer]] = {}
    for bp in board:
        by_pos.setdefault(bp.position, []).append(bp)

    for pos, players in by_pos.items():
        players.sort(key=lambda p: p.season_points, reverse=True)
        # +1.6 bench-ish depth per starting slot demanded league-wide feels right
        # for RB/WR; onesie positions replace at roughly starters + a few.
        slots = starters.get(pos, 0.0)
        if slots <= 0:
            replacement_rank = len(players)
        elif pos in ("QB", "TE", "K", "DST"):
            replacement_rank = round(num_teams * slots) + 3
        else:
            replacement_rank = round(num_teams * slots * 1.45)
        replacement_rank = max(1, min(replacement_rank, len(players)))
        replacement_points = players[replacement_rank - 1].season_points

        tier, prev_points = 1, None
        for rank, bp in enumerate(players, start=1):
            bp.vorp = round(bp.season_points - replacement_points, 1)
            bp.position_rank = rank
            gap_threshold = max(6.0, 0.045 * players[0].season_points)
            if prev_points is not None and prev_points - bp.season_points > gap_threshold:
                tier += 1
            bp.tier = tier
            prev_points = bp.season_points

    board.sort(key=lambda p: p.vorp, reverse=True)
    for i, bp in enumerate(board, start=1):
        bp.overall_rank = i


# ---------------------------------------------------------------------------
# Live draft recommendations
# ---------------------------------------------------------------------------

@dataclass
class RosterNeeds:
    """My roster's unfilled slots, from league settings + players I've drafted."""

    starters_needed: dict[str, float]
    flex_needed: float
    bench_open: int
    kd_slots_open: int  # K + DST starting slots still unfilled
    total_open: int


def compute_needs(
    roster_positions: list[dict], my_positions: list[str]
) -> RosterNeeds:
    dedicated: dict[str, int] = {}
    flex_slots = 0
    bench = 0
    for slot in roster_positions:
        pos, count = slot.get("position"), slot.get("count", 0)
        if pos in ("BN",):
            bench += count
        elif pos in ("IR", "IR+"):
            continue
        elif pos in FLEX_SHARES:
            flex_slots += count
        elif pos:
            key = "DST" if pos == "DEF" else pos
            dedicated[key] = dedicated.get(key, 0) + count

    have: dict[str, int] = {}
    for p in my_positions:
        key = "DST" if p == "DEF" else p
        have[key] = have.get(key, 0) + 1

    starters_needed: dict[str, float] = {}
    surplus: dict[str, int] = {}
    for pos, need in dedicated.items():
        got = have.get(pos, 0)
        starters_needed[pos] = max(0, need - got)
        surplus[pos] = max(0, got - need)

    flex_eligible_surplus = sum(surplus.get(p, 0) for p in ("RB", "WR", "TE"))
    flex_needed = max(0, flex_slots - flex_eligible_surplus)

    total_slots = sum(
        s.get("count", 0) for s in roster_positions if s.get("position") not in ("IR", "IR+")
    )
    total_open = max(0, total_slots - len(my_positions))
    kd_open = int(starters_needed.get("K", 0) + starters_needed.get("DST", 0))
    bench_open = max(0, total_open - int(sum(starters_needed.values())) - flex_needed)

    return RosterNeeds(
        starters_needed=starters_needed,
        flex_needed=flex_needed,
        bench_open=bench_open,
        kd_slots_open=kd_open,
        total_open=total_open,
    )


def recommend(
    board: list[BoardPlayer],
    taken_norm_names: set[tuple[str, str]],
    taken_dst_teams: set[str],
    my_positions: list[str],
    roster_positions: list[dict],
    picks_until_next: int,
    top_n: int = 8,
) -> list[dict]:
    """Rank the best available players for MY next pick, with human-readable reasons."""
    needs = compute_needs(roster_positions, my_positions)

    available = [
        bp
        for bp in board
        if (bp.norm_name, bp.position) not in taken_norm_names
        and not (bp.position == "DST" and (bp.nfl_team or "") in taken_dst_teams)
    ]

    remaining_in_tier: dict[tuple[str, int], int] = {}
    for bp in available:
        key = (bp.position, bp.tier)
        remaining_in_tier[key] = remaining_in_tier.get(key, 0) + 1

    my_remaining_picks = needs.total_open
    scored: list[tuple[float, BoardPlayer, list[str]]] = []
    for bp in available:
        reasons: list[str] = []
        pos = bp.position

        if pos in ("K", "DST"):
            # Only draft K/DST when the roster is nearly full — every earlier
            # pick is worth more as a skill player.
            if my_remaining_picks > needs.kd_slots_open + 1 or needs.kd_slots_open == 0:
                continue
            need_mult = 1.0 if needs.starters_needed.get(pos, 0) > 0 else 0.0
            if need_mult:
                reasons.append(f"Time to fill your {pos} slot")
        else:
            if needs.starters_needed.get(pos, 0) > 0:
                need_mult = 1.0
                reasons.append(f"{pos} starter still needed")
            elif needs.flex_needed > 0 and pos in ("RB", "WR", "TE"):
                need_mult = 0.85
                reasons.append("Fills a flex spot")
            elif needs.bench_open > 0:
                # Depth QBs/TEs are worth much less than depth RB/WRs.
                need_mult = 0.35 if pos in ("QB", "TE") else 0.6
                reasons.append("Bench depth")
            else:
                need_mult = 0.1

        score = bp.vorp * need_mult if bp.vorp > 0 else bp.vorp

        tier_left = remaining_in_tier.get((pos, bp.tier), 0)
        if (
            pos not in ("K", "DST")
            and bp.vorp > 0
            and tier_left <= max(2, picks_until_next // 4)
            and picks_until_next > 0
        ):
            score *= 1.15
            reasons.append(
                f"Tier {bp.tier} {pos} is nearly gone — {tier_left} left, "
                f"{picks_until_next} picks before your next turn"
            )

        if bp.source == "market":
            reasons.append(f"Market-valued (ADP {bp.adp:.1f}) — no model history (likely rookie)")
        elif bp.adp is not None and bp.overall_rank + 18 < bp.adp:
            reasons.append(f"Model likes him more than the market (ADP {bp.adp:.1f})")

        if bp.injury_status:
            score *= 0.9
            reasons.append(f"Carries an injury tag ({bp.injury_status})")

        if not reasons or reasons[-1] != "Bench depth":
            reasons.insert(
                0,
                f"{bp.vorp:+.1f} pts over replacement ({bp.season_points:.0f} projected season pts)",
            )
        scored.append((score, bp, reasons))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [
        {
            "name": bp.name,
            "position": bp.position,
            "nfl_team": bp.nfl_team,
            "score": round(score, 1),
            "vorp": bp.vorp,
            "season_points": round(bp.season_points, 1),
            "adp": bp.adp,
            "tier": bp.tier,
            "overall_rank": bp.overall_rank,
            "position_rank": bp.position_rank,
            "source": bp.source,
            "injury_status": bp.injury_status,
            "reasons": reasons[:3],
        }
        for score, bp, reasons in scored[:top_n]
    ]


def match_taken(
    board: list[BoardPlayer], drafted_players: list[dict]
) -> tuple[set[tuple[str, str]], set[str]]:
    """Convert Yahoo drafted-player info into the board's (norm_name, position)
    key set + taken DST team abbrs. Yahoo 'DEF' maps to our 'DST'."""
    taken: set[tuple[str, str]] = set()
    taken_dst: set[str] = set()
    board_names = {bp.norm_name for bp in board}
    for p in drafted_players:
        pos = p.get("primary_position") or p.get("position") or ""
        if pos == "DEF":
            taken_dst.add((p.get("nfl_team") or "").upper())
            continue
        norm = normalize_name(p.get("name") or "")
        if norm in board_names:
            taken.add((norm, pos))
        else:
            # Fall back to name-only marking across positions (handles Yahoo
            # position quirks like "WR,TE").
            for bp_pos in ("QB", "RB", "WR", "TE", "K"):
                taken.add((norm, bp_pos))
    return taken, taken_dst
