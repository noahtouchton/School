from typing import List, Dict
from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings
from ...models import Team, TradeProposal


class TradeRefuser(BaseAgent):
    """Won't deal. Flat-out refuses every trade proposal and never initiates one.
    Compensates by working the waiver wire harder than anyone and drafting for depth.
    """
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            # Aggressive waiver to compensate for no trades
            waiver_min_improvement=1.0,
            waiver_max_faab_pct=0.18,
            # Slight depth bias in draft — values bench players more
            vorp_decay_rb=0.5,
            vorp_decay_wr=0.5,
        ))

    def generate_trade_proposals(self, team: Team, all_teams: List[Team],
                                 projs: Dict[str, float]) -> List[TradeProposal]:
        return []

    def evaluate_trade_proposal(self, team: Team, proposal: TradeProposal,
                                projs: Dict[str, float]) -> bool:
        return False
