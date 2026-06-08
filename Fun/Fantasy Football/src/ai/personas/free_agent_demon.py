from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings

class FreeAgentDemon(BaseAgent):
    """The Free Agent Demon manager, represented by aggressive waiver wire parameters."""
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            waiver_min_improvement=0.5,
            waiver_max_faab_pct=0.20
        ))
