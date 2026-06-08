from typing import List, Dict, Any, Tuple
from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings
from ...models import Player, Roster
from ..optimizer import solve_optimal_lineup


class StackBuilder(BaseAgent):
    """Builds NFL team stacks. Drafts a QB reasonably early, then aggressively targets
    that QB's WRs and TE to maximize correlated upside. Starts the stack every week.
    """

    STACK_BOOST = 2.5  # VORP multiplier applied to same-team WR/TE after QB is drafted

    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            # Must secure a QB in rounds 3–6 to build around
            early_qb_limit=1,
            vorp_decay_qb=0.3,  # Very low — only want one QB
            vorp_decay_wr=0.55,
            vorp_decay_te=0.45,
            trade_min_gain=1.5,
            waiver_min_improvement=2.0,
            waiver_max_faab_pct=0.08,
            matchup_adjustment=0.2,
        ))

    def _get_stacked_nfl_team(self, roster_players: List[Player]) -> str:
        """Returns the NFL team of the first QB on the roster, or empty string."""
        for p in roster_players:
            pos_str = p.position.value if hasattr(p.position, "value") else str(p.position)
            if pos_str == "QB":
                return p.nfl_team
        return ""

    def draft_pick(self, draft_state: Any, available_players: List[Player],
                   projs: Dict[str, float]) -> Player:
        """Applies a VORP boost to WRs and TEs who share an NFL team with the drafted QB."""
        team = [t for t in draft_state.teams if t.id == self.team_id][0]
        vorp = self.get_vorp_scores(available_players, projs)

        roster_players = team.roster.all_players()
        stacked_team = self._get_stacked_nfl_team(roster_players)

        qb_drafted = sum(1 for p in roster_players if (p.position.value if hasattr(p.position, "value") else str(p.position)) == "QB")
        rb_drafted = sum(1 for p in roster_players if (p.position.value if hasattr(p.position, "value") else str(p.position)) == "RB")
        wr_drafted = sum(1 for p in roster_players if (p.position.value if hasattr(p.position, "value") else str(p.position)) == "WR")
        te_drafted = sum(1 for p in roster_players if (p.position.value if hasattr(p.position, "value") else str(p.position)) == "TE")

        current_round = draft_state.current_round
        total_rounds = self.settings.roster.total_roster_spots()

        adjusted_vorp = {}
        for p in available_players:
            p_vorp = vorp.get(p.id, 0.0)
            pos_str = p.position.value if hasattr(p.position, "value") else str(p.position)

            penalty = 1.0
            if pos_str == "QB" and qb_drafted >= self.settings.roster.qb:
                penalty = self.params.vorp_decay_qb ** (qb_drafted - self.settings.roster.qb + 1)
            elif pos_str == "RB" and rb_drafted >= self.settings.roster.rb:
                penalty = self.params.vorp_decay_rb ** (rb_drafted - self.settings.roster.rb + 1)
            elif pos_str == "WR" and wr_drafted >= self.settings.roster.wr:
                penalty = self.params.vorp_decay_wr ** (wr_drafted - self.settings.roster.wr + 1)
            elif pos_str == "TE" and te_drafted >= self.settings.roster.te:
                penalty = self.params.vorp_decay_te ** (te_drafted - self.settings.roster.te + 1)

            if pos_str == "QB" and current_round <= 6 and qb_drafted >= self.params.early_qb_limit:
                p_vorp = -100.0

            if pos_str in ["K", "DST"] and current_round < total_rounds - 2:
                p_vorp = -100.0

            if p.experience == 0:
                penalty *= self.params.rookie_boost

            if p.age and p.age > self.params.age_penalty_threshold:
                age_excess = p.age - self.params.age_penalty_threshold
                penalty *= max(0.2, 1.0 - (age_excess * self.params.age_penalty_factor))

            # Core stacking logic: boost same-team WR/TE after QB is secured
            if stacked_team and p.nfl_team == stacked_team and pos_str in ["WR", "TE"]:
                penalty *= self.STACK_BOOST

            adjusted_vorp[p.id] = p_vorp * penalty

        sorted_by_vorp = sorted(available_players, key=lambda p: adjusted_vorp.get(p.id, -999.0), reverse=True)
        return sorted_by_vorp[0]

    def optimize_weekly_lineup(self, roster: Roster,
                               projections: Dict[str, float]) -> Tuple[List[Player], List[Player]]:
        """Applies a small starting boost to stacked players to prefer starting the correlation."""
        stacked_team = self._get_stacked_nfl_team(roster.all_players())
        boosted = dict(projections)
        if stacked_team:
            for p in roster.all_players():
                pos_str = p.position.value if hasattr(p.position, "value") else str(p.position)
                if p.nfl_team == stacked_team and pos_str in ["QB", "WR", "TE"]:
                    boosted[p.id] = projections.get(p.id, 0.0) * 1.08
        return solve_optimal_lineup(roster.all_players(), boosted, self.settings.roster)
