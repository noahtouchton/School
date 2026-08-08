from typing import List, Dict, Tuple, Optional, Any
from ..models import Player, Roster, Team, WaiverClaim, TradeProposal
from ..ai.base_agent import BaseAgent
from ..ai.optimizer import solve_optimal_lineup

class ESPNStrategyAdvisor:
    """Uses a trained AI Agent to generate actionable ESPN fantasy football strategies."""
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def analyze_start_sit(self, user_team: Team, projections: Dict[str, float]) -> Dict[str, Any]:
        """Analyzes current starters vs bench and returns recommended lineup swaps."""
        all_players = user_team.roster.all_players()
        current_starters = user_team.roster.starters
        current_bench = user_team.roster.bench
        
        current_proj_total = sum(projections.get(p.id, 0.0) for p in current_starters)
        
        # Calculate optimal lineup using MILP solver
        optimal_starters, optimal_bench = self.agent.optimize_weekly_lineup(user_team.roster, projections)
        optimal_proj_total = sum(projections.get(p.id, 0.0) for p in optimal_starters)
        
        swaps = []
        # Find players on current bench who should START
        to_start = [p for p in optimal_starters if p not in current_starters]
        # Find current starters who should be BENCHED
        to_bench = [p for p in current_starters if p not in optimal_starters]
        
        for p_start, p_bench in zip(to_start, to_bench):
            gain = projections.get(p_start.id, 0.0) - projections.get(p_bench.id, 0.0)
            swaps.append({
                "start_player": p_start,
                "bench_player": p_bench,
                "start_proj": projections.get(p_start.id, 0.0),
                "bench_proj": projections.get(p_bench.id, 0.0),
                "point_gain": round(gain, 2)
            })
            
        return {
            "current_projected": round(current_proj_total, 2),
            "optimal_projected": round(optimal_proj_total, 2),
            "potential_gain": round(max(0.0, optimal_proj_total - current_proj_total), 2),
            "recommended_swaps": swaps,
            "optimal_starters": optimal_starters,
            "optimal_bench": optimal_bench
        }

    def analyze_waivers(self, user_team: Team, free_agents: List[Player], 
                        projections: Dict[str, float], week: int = 1) -> List[Dict[str, Any]]:
        """Scans ESPN free agents and recommends top additions, drops, and FAAB bids."""
        claims = self.agent.get_waiver_claims(user_team, free_agents, projections, current_week=week)
        
        recommendations = []
        for claim in claims:
            add_p = claim.player_to_add
            drop_p = claim.player_to_drop
            add_proj = projections.get(add_p.id, 0.0)
            drop_proj = projections.get(drop_p.id, 0.0) if drop_p else 0.0
            
            recommendations.append({
                "add_player": add_p,
                "drop_player": drop_p,
                "bid_amount": claim.bid_amount,
                "add_proj": add_proj,
                "drop_proj": drop_proj,
                "projected_gain": round(add_proj - drop_proj, 2),
                "reasoning": f"Add {add_p.name} (+{round(add_proj - drop_proj, 1)} pts/wk). Recommended bid: ${claim.bid_amount} FAAB."
            })
            
        return recommendations

    def analyze_trades(self, user_team: Team, all_teams: List[Team], 
                       projections: Dict[str, float]) -> List[Dict[str, Any]]:
        """Scans all ESPN league franchises to find win-win trade offers."""
        proposals = self.agent.generate_trade_proposals(user_team, all_teams, projections)
        
        trade_recs = []
        team_map = {t.id: t for t in all_teams}
        
        for prop in proposals:
            target_team = team_map.get(prop.receiver_team_id)
            if not target_team:
                continue
                
            give_names = ", ".join(p.name for p in prop.proposer_sends)
            recv_names = ", ".join(p.name for p in prop.receiver_sends)
            
            trade_recs.append({
                "target_team": target_team.name,
                "give_players": prop.proposer_sends,
                "receive_players": prop.receiver_sends,
                "summary": f"Send ({give_names}) ➔ Receive ({recv_names}) from {target_team.name}",
                "status": "Recommended"
            })
            
        return trade_recs

    def analyze_draft_picks(self, available_players: List[Player], projections: Dict[str, float], 
                            draft_state: Any, top_n: int = 5) -> List[Dict[str, Any]]:
        """Computes top recommended draft picks using VORP and agent parameters."""
        vorp_scores = self.agent.get_vorp_scores(available_players, projections)
        
        recs = []
        for p in available_players:
            pos_str = p.position.value if hasattr(p.position, "value") else str(p.position)
            proj = projections.get(p.id, 0.0)
            vorp = vorp_scores.get(p.id, 0.0)
            
            recs.append({
                "player": p,
                "position": pos_str,
                "projected_points": proj,
                "vorp": round(vorp, 2)
            })
            
        recs = sorted(recs, key=lambda x: x["vorp"], reverse=True)
        return recs[:top_n]
