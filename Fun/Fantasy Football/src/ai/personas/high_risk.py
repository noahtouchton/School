from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings

class HighRiskAgent(BaseAgent):
    """The High-Risk manager, parameterized to chase young breakouts and speculative gains."""
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            rookie_boost=1.15,
            young_player_trade_boost=0.15,
            waiver_min_improvement=1.0,
            waiver_max_faab_pct=0.12
        ))
