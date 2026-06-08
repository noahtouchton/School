from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings

class RobustRBAgent(BaseAgent):
    """The 'Robust RB' agent forces running back selections in the first 3 rounds of the draft."""
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            early_rb_minimum=3
        ))
