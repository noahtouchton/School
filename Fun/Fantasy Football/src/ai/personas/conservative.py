from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings

class ConservativeAgent(BaseAgent):
    """The Conservative manager, represented by risk-averse behavior parameters."""
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            waiver_min_improvement=4.0,
            waiver_max_faab_pct=0.01,
            trade_min_gain=2.0,
            rookie_boost=0.85,
            age_penalty_threshold=30,
            age_penalty_factor=0.02,
            matchup_adjustment=0.2
        ))
