"""The AI manager: turns a parameter vector into draft, lineup, and waiver decisions.

Every decision routes through self.params so the evolutionary trainer can change
behaviour without touching code. The draft scorer also emits a short reason per
pick, which is what the AI Arena shows -- a pick nobody can explain is useless as
a teaching tool even if it's correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.params import AgentParameters
from app.draft.assistant import BoardPlayer

FLEX_ELIGIBLE = {"RB", "WR", "TE"}
EARLY_ROUND_CUTOFF = 6
FORCED_RB_ROUNDS = 3


@dataclass
class RosterSlots:
    """League roster shape, in the canonical slot vocabulary."""

    starters: dict[str, int] = field(default_factory=dict)  # QB/RB/WR/TE/K/DEF
    flex: int = 0
    bench: int = 0

    @property
    def total(self) -> int:
        return sum(self.starters.values()) + self.flex + self.bench

    @classmethod
    def from_positions(cls, roster_positions: list[dict]) -> "RosterSlots":
        starters: dict[str, int] = {}
        flex = 0
        bench = 0
        for slot in roster_positions:
            position, count = slot.get("position"), int(slot.get("count", 0) or 0)
            if not position or not count:
                continue
            if position == "BN":
                bench += count
            elif position in ("IR", "IR+"):
                continue
            elif position in ("W/R/T", "W/T", "W/R", "Q/W/R/T"):
                flex += count
            else:
                starters["DST" if position == "DEF" else position] = (
                    starters.get("DST" if position == "DEF" else position, 0) + count
                )
        return cls(starters=starters, flex=flex, bench=bench)


@dataclass
class SimTeam:
    """A team inside a simulated season."""

    team_id: str
    name: str
    persona: str
    params: AgentParameters
    roster: list[BoardPlayer] = field(default_factory=list)
    faab: int = 100
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for player in self.roster:
            out[player.position] = out.get(player.position, 0) + 1
        return out


class Agent:
    """Parameter-driven decision maker for one simulated team."""

    def __init__(self, params: AgentParameters, slots: RosterSlots):
        self.params = params
        self.slots = slots

    # ------------------------------------------------------------------
    # Draft
    # ------------------------------------------------------------------

    def _positional_need(self, counts: dict[str, int], position: str) -> float:
        """1.0 while a starting slot is unfilled, then the position's decay weight."""
        needed = self.slots.starters.get(position, 0)
        have = counts.get(position, 0)
        if have < needed:
            return 1.0

        # Starters filled. Flex-eligible positions still have real value while a
        # flex slot is open; onesie positions fall straight to their decay weight.
        flex_filled = sum(
            max(0, counts.get(p, 0) - self.slots.starters.get(p, 0)) for p in FLEX_ELIGIBLE
        )
        if position in FLEX_ELIGIBLE and flex_filled < self.slots.flex:
            return 0.9

        decay = {
            "QB": self.params.vorp_decay_qb,
            "RB": self.params.vorp_decay_rb,
            "WR": self.params.vorp_decay_wr,
            "TE": self.params.vorp_decay_te,
        }.get(position, 0.15)
        # Each extra body past the starting requirement is worth less again.
        surplus = have - needed
        return decay * (decay ** surplus)

    def _allowed(self, player: BoardPlayer, counts: dict[str, int], rnd: int, picks_left: int) -> bool:
        """Hard shape constraints -- what makes a strategy recognisable."""
        position = player.position

        if rnd <= EARLY_ROUND_CUTOFF:
            if position == "RB" and counts.get("RB", 0) >= self.params.early_rb_limit:
                return False
            if position == "QB" and counts.get("QB", 0) >= self.params.early_qb_limit:
                return False

        # Robust-RB style: refuse anything but RB early until the quota is met.
        if rnd <= FORCED_RB_ROUNDS and counts.get("RB", 0) < self.params.early_rb_minimum:
            if position != "RB":
                return False

        # Never spend an early pick on a kicker or defense; and never end the
        # draft without the starters the league requires.
        if position in ("K", "DST"):
            still_needed = sum(
                max(0, self.slots.starters.get(p, 0) - counts.get(p, 0)) for p in ("K", "DST")
            )
            if picks_left > still_needed + 1:
                return False
            if counts.get(position, 0) >= self.slots.starters.get(position, 0):
                return False
        return True

    def score_pick(
        self,
        player: BoardPlayer,
        counts: dict[str, int],
        rnd: int,
        picks_until_next: int,
        tier_remaining: int,
        drafted_qb_teams: set[str],
        bye_counts: dict[str, int],
    ) -> tuple[float, list[str]]:
        """Score one candidate, with reasons. Higher is better."""
        reasons: list[str] = []
        need = self._positional_need(counts, player.position)
        score = player.vorp * need

        if need >= 1.0:
            reasons.append(f"still needs a starting {player.position}")
        elif need >= 0.85:
            reasons.append("fills the flex")
        else:
            reasons.append(f"{player.position} depth")

        # Tier scarcity: value a player more when his tier is about to empty out
        # before the next turn comes around.
        if tier_remaining <= 2 and picks_until_next > 0 and player.vorp > 0:
            bump = 1.0 + 0.15 * self.params.tier_scarcity_weight
            score *= bump
            reasons.append(f"last of tier {player.tier} at {player.position}")

        if player.source == "market":
            score *= self.params.rookie_boost
            if self.params.rookie_boost > 1.05:
                reasons.append("rookie upside")
            elif self.params.rookie_boost < 0.95:
                reasons.append("wary of unproven players")

        if player.age and player.age > self.params.age_penalty_threshold:
            years = player.age - self.params.age_penalty_threshold
            score *= max(0.4, 1.0 - self.params.age_penalty_factor * years)
            if self.params.age_penalty_factor > 0.02:
                reasons.append(f"age {player.age} discount")

        if player.position in ("WR", "TE") and player.nfl_team in drafted_qb_teams:
            score *= self.params.qb_wr_stack_boost
            if self.params.qb_wr_stack_boost > 1.05:
                reasons.append("stacks with his own QB")

        if player.bye_week and self.params.bye_stack_penalty > 0:
            stacked = bye_counts.get(player.bye_week, 0)
            if stacked >= 2:
                score *= max(0.5, 1.0 - self.params.bye_stack_penalty * (stacked - 1))
                reasons.append(f"week {player.bye_week} bye already crowded")

        return score, reasons

    def pick(
        self,
        available: list[BoardPlayer],
        team: SimTeam,
        rnd: int,
        picks_until_next: int,
        tier_counts: dict[tuple[str, int], int],
    ) -> tuple[BoardPlayer | None, list[str]]:
        counts = team.counts()
        picks_left = self.slots.total - len(team.roster)
        drafted_qb_teams = {p.nfl_team for p in team.roster if p.position == "QB" and p.nfl_team}
        bye_counts: dict[str, int] = {}
        for p in team.roster:
            if p.bye_week:
                bye_counts[p.bye_week] = bye_counts.get(p.bye_week, 0) + 1

        best: tuple[float, BoardPlayer, list[str]] | None = None
        # Only the top slice can plausibly win; scoring the entire board every
        # pick is the difference between a snappy sim and a slow one.
        for player in available[:80]:
            if not self._allowed(player, counts, rnd, picks_left):
                continue
            tier_remaining = tier_counts.get((player.position, player.tier), 0)
            score, reasons = self.score_pick(
                player, counts, rnd, picks_until_next, tier_remaining, drafted_qb_teams, bye_counts
            )
            if best is None or score > best[0]:
                best = (score, player, reasons)

        if best is None:
            # Constraints ruled everything out (e.g. roster nearly full and only
            # K/DST left but already rostered). Fall back to best available.
            return (available[0], ["best available"]) if available else (None, [])
        return best[1], best[2]

    # ------------------------------------------------------------------
    # Lineup
    # ------------------------------------------------------------------

    def lineup(
        self,
        roster: list[BoardPlayer],
        expected: dict[str, float],
    ) -> list[BoardPlayer]:
        """Greedy best starting lineup by expected points, most-restrictive slot first."""
        chosen: list[BoardPlayer] = []
        used: set[str] = set()

        def take(eligible: set[str], count: int) -> None:
            candidates = sorted(
                (p for p in roster if p.key not in used and p.position in eligible),
                key=lambda p: -expected.get(p.key, 0.0),
            )
            for player in candidates[:count]:
                used.add(player.key)
                chosen.append(player)

        for position, count in self.slots.starters.items():
            take({position}, count)
        if self.slots.flex:
            take(FLEX_ELIGIBLE, self.slots.flex)
        return chosen

    # ------------------------------------------------------------------
    # Waivers
    # ------------------------------------------------------------------

    def waiver_claim(
        self,
        team: SimTeam,
        free_agents: list[BoardPlayer],
        expected: dict[str, float],
        weeks_left: int,
        total_weeks: int,
    ) -> tuple[BoardPlayer, BoardPlayer, int] | None:
        """Best (add, drop, bid) this week, or None if nothing clears the bar."""
        if not free_agents or not team.roster:
            return None

        best: tuple[float, BoardPlayer, BoardPlayer] | None = None
        for candidate in free_agents[:40]:
            same_position = [p for p in team.roster if p.position == candidate.position]
            if not same_position:
                continue
            weakest = min(same_position, key=lambda p: expected.get(p.key, 0.0))
            gain = expected.get(candidate.key, 0.0) - expected.get(weakest.key, 0.0)
            if best is None or gain > best[0]:
                best = (gain, candidate, weakest)

        if best is None or best[0] < self.params.waiver_min_improvement:
            return None

        gain, add, drop = best
        # Budget burns faster late: an unspent FAAB dollar is worth nothing once
        # the season ends, and faab_urgency_factor controls how much the agent
        # believes that.
        urgency = 1.0 + self.params.faab_urgency_factor * (1.0 - weeks_left / max(1, total_weeks))
        bid = int(round(team.faab * self.params.waiver_max_faab_pct * urgency))
        return add, drop, max(0, min(team.faab, bid))
