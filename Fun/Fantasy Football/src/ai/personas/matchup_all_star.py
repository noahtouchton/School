from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings

class MatchupAllStar(BaseAgent):
    """The Matchup All-Star manager, represented by high matchup-sensitivity parameters."""
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            matchup_adjustment=1.0
        ))
