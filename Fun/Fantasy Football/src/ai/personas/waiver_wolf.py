from typing import List, Dict
from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings
from ...models import Team, Player, WaiverClaim


class WaiverWolf(BaseAgent):
    """Lives on the waiver wire. Swaps roster spots aggressively every week,
    bids high on FAAB, and drafts speculative upside picks expecting to replace them.
    """
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            waiver_min_improvement=0.3,
            waiver_max_faab_pct=0.25,
            # Draft high-upside / rookie picks expecting to churn them
            rookie_boost=1.2,
            trade_min_gain=3.0,
        ))

    def get_waiver_claims(self, team: Team, free_agents: List[Player],
                          projs: Dict[str, float], current_week: int) -> List[WaiverClaim]:
        """Scans all free agents (not just top 5) and aggressively churns the bench."""
        claims = []
        bench_players = team.roster.bench
        if not bench_players:
            return []

        bench_sorted = sorted(bench_players, key=lambda p: projs.get(p.id, 0.0))
        # Scan the top 10 free agents instead of 5
        fa_sorted = sorted(free_agents, key=lambda p: projs.get(p.id, 0.0), reverse=True)

        claim_index = 0
        for fa in fa_sorted[:10]:
            for worst_bench in bench_sorted:
                fa_proj = projs.get(fa.id, 0.0)
                bench_proj = projs.get(worst_bench.id, 0.0)

                adj_fa_proj = fa_proj
                if fa.experience == 0:
                    adj_fa_proj *= self.params.rookie_boost

                if adj_fa_proj - bench_proj > self.params.waiver_min_improvement:
                    import random
                    max_bid = int(self.settings.faab_budget * self.params.waiver_max_faab_pct)
                    bid = 0
                    if team.faab_balance > 0 and max_bid > 0:
                        bid = random.randint(1, min(max_bid, team.faab_balance))

                    claims.append(WaiverClaim(
                        team_id=self.team_id,
                        player_to_add=fa,
                        player_to_drop=worst_bench,
                        bid_amount=bid,
                        priority_order=claim_index
                    ))
                    claim_index += 1
                    break

        return claims
