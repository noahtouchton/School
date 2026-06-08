import random
from typing import List, Dict, Tuple, Optional
from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings
from ...models import Team, Player, TradeProposal
from ..optimizer import solve_optimal_lineup


class BigTrader(BaseAgent):
    """Always looking for a deal. Proposes trades constantly, accepts almost anything
    that marginally improves the roster, and drafts RB-heavy to stockpile trade assets.
    """
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            trade_min_gain=0.05,
            young_player_trade_boost=0.2,
            # Stockpile RBs early as trade chips
            early_rb_minimum=2,
            vorp_decay_rb=0.55,
            waiver_min_improvement=2.5,
            waiver_max_faab_pct=0.05,
        ))

    def generate_trade_proposals(self, team: Team, all_teams: List[Team],
                                 projs: Dict[str, float]) -> List[TradeProposal]:
        """Scans all teams for any trade that marginally improves the starting lineup,
        including same-position upgrades. Proposes up to 3 trades per week.
        """
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
                    my_hyp_roster = [p for p in my_roster if p.id != my_player.id] + [other_player]
                    my_hyp_starters, _ = solve_optimal_lineup(my_hyp_roster, projs, self.settings.roster)
                    my_score_after = sum(projs.get(p.id, 0.0) for p in my_hyp_starters)
                    my_gain = my_score_after - my_score_before

                    other_hyp_roster = [p for p in other_roster if p.id != other_player.id] + [my_player]
                    other_hyp_starters, _ = solve_optimal_lineup(other_hyp_roster, projs, self.settings.roster)
                    other_score_after = sum(projs.get(p.id, 0.0) for p in other_hyp_starters)
                    other_gain = other_score_after - other_score_before

                    # Very low mutual benefit threshold — any deal that moves the needle
                    if my_gain >= self.params.trade_min_gain and other_gain >= 0.05:
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
                if len(proposals) >= 3:
                    break

        return proposals
