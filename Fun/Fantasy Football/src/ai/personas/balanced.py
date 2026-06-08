from ..base_agent import BaseAgent
from ...config import LeagueSettings

class BalancedAgent(BaseAgent):
    """Balanced Agent. Inherits all standard VORP, waiver, and trade logic with no skew."""
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings)
