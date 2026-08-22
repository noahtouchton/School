"""Full-season simulator: draft a league of AI managers, then replay a real NFL season.

Design choice that matters: weekly scores are the players' **actual** historical
fantasy points, not samples from a distribution. Replaying a season that really
happened means a strategy is judged against real injuries, real breakouts and real
busts, which is the only thing that makes the evolutionary trainer's output worth
anything. Invented randomness would just reward whoever matched the generator.

Two rules keep it honest:
  * The draft board is built from seasons strictly BEFORE the replay season, so no
    agent drafts knowing how the year turned out.
  * Weekly lineups are set from form entering that week (a trailing average of games
    already played) -- never from the points about to be scored.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.agent import Agent, RosterSlots, SimTeam
from app.agents.params import AgentParameters
from app.agents.personas import persona_params
from app.core.scoring import calculate_fantasy_points
from app.db.models import Player, PlayerWeeklyStat
from app.draft.assistant import BoardPlayer, apply_vorp, build_board

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

REGULAR_SEASON_WEEKS = 14
# Weight on a player's season-long form vs. his last three games when deciding
# who to start. Recent form matters but overreacting to one game is how managers
# bench the right player.
FORM_RECENT_WEIGHT = 0.4


@dataclass
class DraftPick:
    overall: int
    round: int
    team_id: str
    team_name: str
    player: str
    position: str
    nfl_team: str | None
    reasons: list[str]
    vorp: float


@dataclass
class WeekResult:
    week: int
    matchups: list[dict]
    transactions: list[dict] = field(default_factory=list)


@dataclass
class SeasonResult:
    season: int
    teams: list[dict]
    draft: list[DraftPick]
    weeks: list[WeekResult]
    champion: str | None
    standings: list[dict]


@dataclass
class PreparedSeason:
    """Everything a season replay needs that doesn't depend on the agents.

    Built once and shared across runs -- the board and the stat lookup are pure
    functions of (season, league shape), so recomputing them per candidate is
    wasted work in the training loop.
    """

    season: int
    board: list[BoardPlayer]
    actual: dict[tuple[str, int], float]


def prepare_season(
    db: Session, season: int, roster_positions: list[dict], num_teams: int
) -> PreparedSeason:
    board = build_board(db, through_season=season - 1)
    if not board:
        raise ValueError(f"no draft board available for {season}; ingest more history")
    apply_vorp(board, roster_positions, num_teams)
    actual, played = _actual_points(db, season)
    _bye_weeks(played, board, REGULAR_SEASON_WEEKS)
    return PreparedSeason(season=season, board=board, actual=actual)


def _actual_points(db: Session, season: int) -> tuple[dict[tuple[str, int], float], dict[str, set[int]]]:
    """(player_id, week) -> real fantasy points, plus which weeks each player played."""
    rows = (
        db.execute(select(PlayerWeeklyStat).where(PlayerWeeklyStat.season == season))
        .scalars()
        .all()
    )
    points: dict[tuple[str, int], float] = {}
    played: dict[str, set[int]] = {}
    for row in rows:
        points[(row.player_id, row.week)] = calculate_fantasy_points(row.stats or {})
        played.setdefault(row.player_id, set()).add(row.week)
    return points, played


def _round_robin(team_ids: list[str], weeks: int, rng: random.Random) -> dict[int, list[tuple[str, str]]]:
    """Standard circle-method rotation, repeated until the season is full."""
    ids = list(team_ids)
    rng.shuffle(ids)
    if len(ids) % 2:
        raise ValueError("simulated leagues need an even number of teams")

    schedule: dict[int, list[tuple[str, str]]] = {}
    rotating = list(ids)
    for week in range(1, weeks + 1):
        pairs = []
        half = len(rotating) // 2
        for i in range(half):
            pairs.append((rotating[i], rotating[len(rotating) - 1 - i]))
        schedule[week] = pairs
        rotating = [rotating[0], rotating[-1], *rotating[1:-1]]
    return schedule


def _bye_weeks(played: dict[str, set[int]], board: list[BoardPlayer], season_weeks: int) -> None:
    """Tag each board player's bye: the week his team didn't play.

    Derived from the stat record rather than the schedule table -- a player with
    no row in exactly one mid-season week is on bye. Ambiguous cases (injured,
    benched, multiple gaps) are left as None rather than guessed at.
    """
    for player in board:
        if not player.player_id:
            continue
        weeks = played.get(player.player_id)
        if not weeks:
            continue
        first, last = min(weeks), max(weeks)
        if last - first < 6:
            continue  # too little of the season to infer anything
        gaps = [w for w in range(first, last + 1) if w not in weeks]
        if len(gaps) == 1:
            player.bye_week = gaps[0]


def simulate_season(
    db: Session,
    season: int,
    candidate: AgentParameters | None = None,
    opponents: list[str] | None = None,
    roster_positions: list[dict] | None = None,
    seed: int = 0,
    capture_detail: bool = True,
    prepared: "PreparedSeason | None" = None,
) -> SeasonResult:
    """Draft and play one full season.

    candidate, when given, is team 1 and the rest are personas -- that's the shape
    the evolutionary trainer uses. With no candidate every team is a persona.

    prepared reuses an already-built board and stat lookup. The evolutionary
    trainer runs hundreds of seasons over identical inputs, and rebuilding those
    each time costs far more than the simulation itself.
    """
    rng = random.Random(seed)
    roster_positions = roster_positions or DEFAULT_ROSTER_POSITIONS
    slots = RosterSlots.from_positions(roster_positions)
    opponents = opponents or []
    num_teams = len(opponents) + (1 if candidate is not None else 0)
    if num_teams < 2:
        raise ValueError("need at least two teams")

    if prepared is None:
        prepared = prepare_season(db, season, roster_positions, num_teams)
    board, actual = prepared.board, prepared.actual

    # --- Teams.
    teams: list[SimTeam] = []
    agents: dict[str, Agent] = {}
    if candidate is not None:
        teams.append(
            SimTeam(team_id="candidate", name="Trained Agent", persona="trained", params=candidate)
        )
    for i, persona in enumerate(opponents):
        team_id = f"p{i}"
        teams.append(
            SimTeam(
                team_id=team_id,
                name=f"{persona.replace('_', ' ').title()}",
                persona=persona,
                params=persona_params(persona),
            )
        )
    for team in teams:
        agents[team.team_id] = Agent(team.params, slots)

    # --- Snake draft.
    order = [t.team_id for t in teams]
    rng.shuffle(order)
    by_id = {t.team_id: t for t in teams}
    available = list(board)
    tier_counts: dict[tuple[str, int], int] = {}
    for p in available:
        tier_counts[(p.position, p.tier)] = tier_counts.get((p.position, p.tier), 0) + 1

    draft_log: list[DraftPick] = []
    overall = 0
    for rnd in range(1, slots.total + 1):
        sequence = order if rnd % 2 == 1 else list(reversed(order))
        for team_id in sequence:
            overall += 1
            team = by_id[team_id]
            agent = agents[team_id]
            picks_until_next = len(order)
            player, reasons = agent.pick(available, team, rnd, picks_until_next, tier_counts)
            if player is None:
                continue
            available.remove(player)
            tier_counts[(player.position, player.tier)] = max(
                0, tier_counts.get((player.position, player.tier), 1) - 1
            )
            team.roster.append(player)
            if capture_detail:
                draft_log.append(
                    DraftPick(
                        overall=overall,
                        round=rnd,
                        team_id=team_id,
                        team_name=team.name,
                        player=player.name,
                        position=player.position,
                        nfl_team=player.nfl_team,
                        reasons=reasons,
                        vorp=player.vorp,
                    )
                )

    # --- Season.
    schedule = _round_robin([t.team_id for t in teams], REGULAR_SEASON_WEEKS, rng)
    free_agents = available
    week_results: list[WeekResult] = []

    # Running form per player, built only from weeks already played.
    history: dict[str, list[float]] = {}

    def expected_points(player: BoardPlayer, week: int) -> float:
        """What a manager could reasonably expect, knowing only prior weeks."""
        if player.bye_week == week:
            return 0.0
        seen = history.get(player.key, [])
        prior = player.per_game if player.per_game is not None else player.season_points / 16.0
        if not seen:
            return prior
        recent = sum(seen[-3:]) / len(seen[-3:])
        season_avg = sum(seen) / len(seen)
        form = FORM_RECENT_WEIGHT * recent + (1 - FORM_RECENT_WEIGHT) * season_avg
        # Blend toward preseason expectation early, when the sample is tiny.
        confidence = min(1.0, len(seen) / 6.0)
        return confidence * form + (1 - confidence) * prior

    for week in range(1, REGULAR_SEASON_WEEKS + 1):
        transactions: list[dict] = []

        # Waivers run before lineups, worst record first.
        claim_order = sorted(teams, key=lambda t: (t.wins, t.points_for))
        for team in claim_order:
            expected = {
                p.key: expected_points(p, week) for p in team.roster + free_agents[:40]
            }
            claim = agents[team.team_id].waiver_claim(
                team, free_agents, expected, REGULAR_SEASON_WEEKS - week, REGULAR_SEASON_WEEKS
            )
            if claim is None:
                continue
            add, drop, bid = claim
            if add not in free_agents:
                continue
            free_agents.remove(add)
            team.roster.remove(drop)
            team.roster.append(add)
            free_agents.append(drop)
            team.faab -= bid
            if capture_detail:
                transactions.append(
                    {
                        "team": team.name,
                        "added": add.name,
                        "dropped": drop.name,
                        "bid": bid,
                    }
                )

        # Lineups, then score against what actually happened.
        scores: dict[str, float] = {}
        lineups: dict[str, list[BoardPlayer]] = {}
        for team in teams:
            expected = {p.key: expected_points(p, week) for p in team.roster}
            starters = agents[team.team_id].lineup(team.roster, expected)
            lineups[team.team_id] = starters
            scores[team.team_id] = round(
                sum(actual.get((p.player_id, week), 0.0) for p in starters if p.player_id), 2
            )

        # Update form with this week's real results (every rostered player, so a
        # benched player's form still moves).
        for team in teams:
            for player in team.roster:
                if player.player_id and (player.player_id, week) in actual:
                    history.setdefault(player.key, []).append(actual[(player.player_id, week)])

        matchups = []
        for home_id, away_id in schedule[week]:
            home, away = by_id[home_id], by_id[away_id]
            home_score, away_score = scores[home_id], scores[away_id]
            home.points_for += home_score
            home.points_against += away_score
            away.points_for += away_score
            away.points_against += home_score
            if home_score > away_score:
                home.wins += 1
                away.losses += 1
            elif away_score > home_score:
                away.wins += 1
                home.losses += 1
            else:
                home.ties += 1
                away.ties += 1
            if capture_detail:
                matchups.append(
                    {
                        "home": home.name,
                        "away": away.name,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_starters": [
                            {
                                "name": p.name,
                                "position": p.position,
                                "points": round(actual.get((p.player_id, week), 0.0), 2),
                            }
                            for p in lineups[home_id]
                        ],
                        "away_starters": [
                            {
                                "name": p.name,
                                "position": p.position,
                                "points": round(actual.get((p.player_id, week), 0.0), 2),
                            }
                            for p in lineups[away_id]
                        ],
                    }
                )
        if capture_detail:
            week_results.append(WeekResult(week=week, matchups=matchups, transactions=transactions))

    def standings_key(t: SimTeam):
        games = t.wins + t.losses + t.ties
        win_pct = (t.wins + 0.5 * t.ties) / games if games else 0.0
        return (win_pct, t.points_for)

    ranked = sorted(teams, key=standings_key, reverse=True)
    standings = [
        {
            "rank": i + 1,
            "team_id": t.team_id,
            "name": t.name,
            "persona": t.persona,
            "wins": t.wins,
            "losses": t.losses,
            "ties": t.ties,
            "points_for": round(t.points_for, 1),
            "points_against": round(t.points_against, 1),
            "faab_left": t.faab,
        }
        for i, t in enumerate(ranked)
    ]

    return SeasonResult(
        season=season,
        teams=[
            {
                "team_id": t.team_id,
                "name": t.name,
                "persona": t.persona,
                "roster": [
                    {"name": p.name, "position": p.position, "nfl_team": p.nfl_team}
                    for p in t.roster
                ],
            }
            for t in teams
        ],
        draft=draft_log,
        weeks=week_results,
        champion=ranked[0].name if ranked else None,
        standings=standings,
    )


def fitness(result: SeasonResult, team_id: str = "candidate") -> float:
    """Wins dominate, points break ties -- the original trainer's objective.

    Points-for alone would reward a team that blows out opponents while losing
    close games; wins alone is far too coarse a signal over 14 games.
    """
    for row in result.standings:
        if row["team_id"] == team_id:
            return row["wins"] * 100.0 + row["points_for"]
    return 0.0
