from typing import List, Dict
from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings
from ...models import Team, Player, WaiverClaim


class RookieHunter(BaseAgent):
    """Obsessed with youth. Aggressively drafts rookies and young players, applies steep
    age penalties to veterans, and targets young free agents on the waiver wire.
    Trades for young players at a premium.
    """
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            rookie_boost=1.6,
            age_penalty_threshold=27,
            age_penalty_factor=0.12,
            young_player_trade_boost=0.6,
            trade_min_gain=0.8,
            waiver_min_improvement=1.8,
            waiver_max_faab_pct=0.15,
        ))

    def get_waiver_claims(self, team: Team, free_agents: List[Player],
                          projs: Dict[str, float], current_week: int) -> List[WaiverClaim]:
        """Prefers rookie/young free agents, applying the rookie boost to waiver decisions."""
        import random
        claims = []
        bench_players = team.roster.bench
        if not bench_players:
            return []

        bench_sorted = sorted(bench_players, key=lambda p: projs.get(p.id, 0.0))
        fa_sorted = sorted(free_agents, key=lambda p: projs.get(p.id, 0.0), reverse=True)

        # Separately bubble up rookie/young FAs to the front of the queue
        young_fas = [p for p in fa_sorted if (p.experience is not None and p.experience <= 1) or (p.age is not None and p.age <= 23)]
        other_fas = [p for p in fa_sorted if p not in young_fas]
        prioritized_fas = young_fas + other_fas

        claim_index = 0
        for fa in prioritized_fas[:8]:
            for worst_bench in bench_sorted:
                fa_proj = projs.get(fa.id, 0.0)
                bench_proj = projs.get(worst_bench.id, 0.0)

                adj_fa_proj = fa_proj
                if fa.experience is not None and fa.experience <= 1:
                    adj_fa_proj *= self.params.rookie_boost

                if adj_fa_proj - bench_proj > self.params.waiver_min_improvement:
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
