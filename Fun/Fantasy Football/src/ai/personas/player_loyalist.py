from typing import List, Dict
from ..base_agent import BaseAgent, AgentParameters
from ...config import LeagueSettings
from ...models import Team, Player, TradeProposal, WaiverClaim
from ..optimizer import solve_optimal_lineup


class PlayerLoyalist(BaseAgent):
    """Believes in their guys. Drafted players stay until the bitter end.
    Almost never trades away a starter and only touches the waiver wire for genuine emergencies.
    Drafts by chasing the best player available at each tier, building a tight core.
    """
    def __init__(self, team_id: str, settings: LeagueSettings):
        super().__init__(team_id, settings, AgentParameters(
            trade_min_gain=7.0,
            waiver_min_improvement=6.0,
            waiver_max_faab_pct=0.04,
            # Values young players they draft and holds them long-term
            young_player_trade_boost=0.4,
            rookie_boost=1.1,
        ))

    def evaluate_trade_proposal(self, team: Team, proposal: TradeProposal,
                                projs: Dict[str, float]) -> bool:
        """Refuses to trade away any top-4 projected starter regardless of the offer."""
        my_roster = team.roster.all_players()
        my_starters, _ = solve_optimal_lineup(my_roster, projs, self.settings.roster)
        # Identify the top 4 starters as "core" players never to be traded
        core = sorted(my_starters, key=lambda p: projs.get(p.id, 0.0), reverse=True)[:4]
        core_ids = {p.id for p in core}

        if proposal.receiver_team_id == self.team_id:
            giving_away = proposal.receiver_sends
        else:
            giving_away = proposal.proposer_sends

        if any(p.id in core_ids for p in giving_away):
            return False

        # For non-core players apply the normal (high) threshold
        return super().evaluate_trade_proposal(team, proposal, projs)

    def get_waiver_claims(self, team: Team, free_agents: List[Player],
                          projs: Dict[str, float], current_week: int) -> List[WaiverClaim]:
        """Only claims a free agent if the bench player being dropped is genuinely irrelevant
        (projected under 4 points) — never drops a productive player.
        """
        claims = []
        bench_players = team.roster.bench
        if not bench_players:
            return []

        # Only consider dropping the very worst bench players
        droppable = [p for p in bench_players if projs.get(p.id, 0.0) < 4.0]
        if not droppable:
            return []

        droppable_sorted = sorted(droppable, key=lambda p: projs.get(p.id, 0.0))
        fa_sorted = sorted(free_agents, key=lambda p: projs.get(p.id, 0.0), reverse=True)

        claim_index = 0
        for fa in fa_sorted[:5]:
            for worst in droppable_sorted:
                fa_proj = projs.get(fa.id, 0.0)
                bench_proj = projs.get(worst.id, 0.0)

                if fa_proj - bench_proj > self.params.waiver_min_improvement:
                    claims.append(WaiverClaim(
                        team_id=self.team_id,
                        player_to_add=fa,
                        player_to_drop=worst,
                        bid_amount=0,
                        priority_order=claim_index
                    ))
                    claim_index += 1
                    break

        return claims
