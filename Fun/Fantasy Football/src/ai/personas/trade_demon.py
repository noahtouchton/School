from typing import List, Dict, Tuple, Optional
from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings
from ...models import Team, Player, TradeProposal
from ..optimizer import solve_optimal_lineup

class TradeDemon(BaseAgent):
    """The Trade Demon manager, driven by highly collaborative trading parameters."""
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            trade_min_gain=0.1
        ))

    def generate_trade_proposals(self, team: Team, all_teams: List[Team], 
                                 projs: Dict[str, float]) -> List[TradeProposal]:
        """Scans other teams' rosters and generates win-win trade proposals."""
        proposals = []
        my_roster = team.roster.all_players()
        my_starters, _ = solve_optimal_lineup(my_roster, projs, self.settings.roster)
        my_score_before = sum(projs.get(p.id, 0.0) for p in my_starters)
        
        for other_team in all_teams:
            if other_team.id == self.team_id:
                continue
                
            other_roster = other_team.roster.all_players()
            other_starters, _ = solve_optimal_lineup(other_roster, projs, self.settings.roster)
            other_score_before = sum(projs.get(p.id, 0.0) for p in other_starters)
            
            best_swap: Optional[Tuple[Player, Player]] = None
            best_my_gain = 0.0
            
            for my_player in my_roster:
                for other_player in other_roster:
                    # Swapping players of different positions to balance rosters
                    if my_player.position == other_player.position:
                        continue
                        
                    # Calculate my score after swap
                    my_hyp_roster = [p for p in my_roster if p.id != my_player.id] + [other_player]
                    my_hyp_starters, _ = solve_optimal_lineup(my_hyp_roster, projs, self.settings.roster)
                    my_score_after = sum(projs.get(p.id, 0.0) for p in my_hyp_starters)
                    my_gain = my_score_after - my_score_before
                    
                    # Calculate other team's score after swap
                    other_hyp_roster = [p for p in other_roster if p.id != other_player.id] + [my_player]
                    other_hyp_starters, _ = solve_optimal_lineup(other_hyp_roster, projs, self.settings.roster)
                    other_score_after = sum(projs.get(p.id, 0.0) for p in other_hyp_starters)
                    other_gain = other_score_after - other_score_before
                    
                    # Propose if BOTH benefit (+0.2 minimum benefit)
                    if my_gain >= 0.2 and other_gain >= 0.2:
                        if my_gain > best_my_gain:
                            best_my_gain = my_gain
                            best_swap = (my_player, other_player)
                            
            if best_swap:
                my_give, other_give = best_swap
                proposals.append(TradeProposal(
                    id=f"trade_{self.team_id}_{other_team.id}_{my_give.id[:4]}",
                    proposer_team_id=self.team_id,
                    receiver_team_id=other_team.id,
                    proposer_sends=[my_give],
                    receiver_sends=[other_give]
                ))
                if len(proposals) >= 2:
                    break
                    
        return proposals
