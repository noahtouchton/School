from typing import List, Dict, Tuple
from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings
from ...models import Roster, Player
from ..optimizer import solve_optimal_lineup


class ProjectionTruster(BaseAgent):
    """The numbers guy. Starts whoever the projections say to start, no exceptions.
    Ignores matchups, age, or gut feel. Drafts by pure VORP rank with no positional bias.
    """
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            matchup_adjustment=0.0,
            # No rookie hype, no age bias — just the numbers
            rookie_boost=1.0,
            age_penalty_factor=0.0,
            # Moderate trade / waiver thresholds driven purely by projected point gain
            trade_min_gain=2.0,
            waiver_min_improvement=2.0,
            waiver_max_faab_pct=0.08,
            # Balanced positional decay so VORP drives picks cleanly
            vorp_decay_qb=0.5,
            vorp_decay_rb=0.4,
            vorp_decay_wr=0.4,
            vorp_decay_te=0.3,
        ))

    def optimize_weekly_lineup(self, roster: Roster,
                               projections: Dict[str, float]) -> Tuple[List[Player], List[Player]]:
        """Always starts the mathematically optimal lineup — never overrides for matchup."""
        return solve_optimal_lineup(roster.all_players(), projections, self.settings.roster)
