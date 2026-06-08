from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings

class HeroRBAgent(BaseAgent):
    """The Hero-RB manager, parameterized to restrict drafting to exactly 1 RB in early rounds."""
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            early_rb_limit=1
        ))
