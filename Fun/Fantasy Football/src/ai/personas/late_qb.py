from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings

class LateRoundQBAgent(BaseAgent):
    """The 'Late Round QB' agent ignores quarterbacks in the first 6 rounds of the draft."""
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            early_qb_limit=0
        ))
