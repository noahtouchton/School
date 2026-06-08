from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings

class ZeroRBAgent(BaseAgent):
    """The Zero-RB manager, parameterized to restrict early drafting of Running Backs."""
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            early_rb_limit=0
        ))
