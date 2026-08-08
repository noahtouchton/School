from typing import Optional
from .base_agent import BaseAgent, AgentParameters
from ..config import LeagueSettings

# Tuned optimal parameter configuration for the Pro AI Engine
PRO_AI_PARAMETERS = AgentParameters(
    vorp_decay_qb=0.65,
    vorp_decay_rb=0.45,
    vorp_decay_wr=0.45,
    vorp_decay_te=0.35,
    early_rb_limit=99,
    early_qb_limit=99,
    early_rb_minimum=0,
    rookie_boost=1.12,
    age_penalty_threshold=29,
    age_penalty_factor=0.03,
    waiver_min_improvement=1.2,
    waiver_max_faab_pct=0.15,
    faab_urgency_factor=1.25,
    trade_min_gain=0.8,
    young_player_trade_boost=0.15,
    trade_future_discount=0.88,
    matchup_adjustment=0.75,
    qb_wr_stack_boost=1.20
)

class ProAIEngine(BaseAgent):
    """The single, state-of-the-art Pro AI Engine.
    Used for all AI draft choices, weekly starting lineups, waiver claims, and trade proposals.
    """
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id=team_id, settings=settings, params=PRO_AI_PARAMETERS)

def get_pro_ai_agent(team_id: str, settings: LeagueSettings) -> ProAIEngine:
    """Factory helper to instantiate the single Pro AI Engine."""
    return ProAIEngine(team_id=team_id, settings=settings)
