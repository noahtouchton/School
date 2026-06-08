from typing import List, Dict, Any
from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings
from ...models import Player
from ..optimizer import solve_optimal_lineup


class WeirdBench(BaseAgent):
    """Ignores conventional roster construction wisdom. Drafts 2 QBs in the first 6 rounds
    for streaming flexibility, picks up K/DST earlier than everyone else, and builds
    a deep non-standard bench others wouldn't consider.
    """
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            # Values QB depth much more than standard
            early_qb_limit=2,
            vorp_decay_qb=0.75,
            # Less emphasis on RB depth
            vorp_decay_rb=0.3,
            vorp_decay_wr=0.35,
            waiver_min_improvement=1.5,
            waiver_max_faab_pct=0.10,
            trade_min_gain=2.0,
            matchup_adjustment=0.5,
        ))

    def draft_pick(self, draft_state: Any, available_players: List[Player],
                   projs: Dict[str, float]) -> Player:
        """Same as base VORP logic but K/DST are no longer exiled to the final 2 rounds.
        Allows drafting a kicker or defense up to 4 rounds early.
        """
        team = [t for t in draft_state.teams if t.id == self.team_id][0]
        vorp = self.get_vorp_scores(available_players, projs)

        roster_players = team.roster.all_players()
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

            if pos_str == "RB" and current_round <= 3 and rb_drafted < self.params.early_rb_minimum:
                penalty *= 2.5

            if p.experience == 0:
                penalty *= self.params.rookie_boost

            if p.age and p.age > self.params.age_penalty_threshold:
                age_excess = p.age - self.params.age_penalty_threshold
                penalty *= max(0.2, 1.0 - (age_excess * self.params.age_penalty_factor))

            # K and DST can be drafted 4 rounds early (not exiled all the way to the end)
            if pos_str in ["K", "DST"] and current_round < total_rounds - 6:
                p_vorp = -100.0

            adjusted_vorp[p.id] = p_vorp * penalty

        sorted_by_vorp = sorted(available_players, key=lambda p: adjusted_vorp.get(p.id, -999.0), reverse=True)
        return sorted_by_vorp[0]
