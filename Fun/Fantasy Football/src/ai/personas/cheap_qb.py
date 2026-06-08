from typing import List, Dict
from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings
from ...models import Team, Player, WaiverClaim


class CheapQB(BaseAgent):
    """Skips QB in the first 7 rounds entirely. Loads up on elite RBs, WRs, and TEs
    then streams QBs off the waiver wire throughout the season.
    """
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            early_qb_limit=0,
            # More investment in skill position depth
            vorp_decay_rb=0.55,
            vorp_decay_wr=0.55,
            vorp_decay_te=0.4,
            # QB stash on waiver is critical — always willing to stream
            waiver_min_improvement=1.2,
            waiver_max_faab_pct=0.12,
            trade_min_gain=2.0,
        ))

    def get_waiver_claims(self, team: Team, free_agents: List[Player],
                          projs: Dict[str, float], current_week: int) -> List[WaiverClaim]:
        """Prioritizes streaming QB adds above all else. Will drop a bench skill player
        to pick up a better QB matchup even at a small gain.
        """
        import random
        claims = []
        bench_players = team.roster.bench
        if not bench_players:
            return []

        bench_sorted = sorted(bench_players, key=lambda p: projs.get(p.id, 0.0))
        fa_sorted = sorted(free_agents, key=lambda p: projs.get(p.id, 0.0), reverse=True)

        claim_index = 0

        # First pass: prioritize QB upgrades with a lower improvement threshold
        for fa in fa_sorted[:10]:
            pos_str = fa.position.value if hasattr(fa.position, "value") else str(fa.position)
            if pos_str != "QB":
                continue
            for worst_bench in bench_sorted:
                fa_proj = projs.get(fa.id, 0.0)
                bench_proj = projs.get(worst_bench.id, 0.0)
                if fa_proj - bench_proj > 0.5:
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

        # Second pass: standard waiver logic for other positions
        for fa in fa_sorted[:5]:
            pos_str = fa.position.value if hasattr(fa.position, "value") else str(fa.position)
            if pos_str == "QB":
                continue
            for worst_bench in bench_sorted:
                fa_proj = projs.get(fa.id, 0.0)
                bench_proj = projs.get(worst_bench.id, 0.0)
                if fa_proj - bench_proj > self.params.waiver_min_improvement:
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
