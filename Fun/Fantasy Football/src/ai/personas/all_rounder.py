from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings


class AllRounder(BaseAgent):
    """The complete manager. Solid VORP drafting, sensible waiver activity,
    fair trade evaluation, and slight matchup awareness. No glaring weaknesses.
    """
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            vorp_decay_qb=0.5,
            vorp_decay_rb=0.4,
            vorp_decay_wr=0.4,
            vorp_decay_te=0.3,
            early_rb_limit=99,
            early_qb_limit=99,
            early_rb_minimum=0,
            rookie_boost=1.05,
            age_penalty_threshold=31,
            age_penalty_factor=0.03,
            waiver_min_improvement=2.0,
            waiver_max_faab_pct=0.08,
            trade_min_gain=1.5,
            young_player_trade_boost=0.1,
            matchup_adjustment=0.3,
        ))
