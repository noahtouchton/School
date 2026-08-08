import random
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from ..config import LeagueSettings, RosterSettings
from ..models import Player, Roster, Team, WaiverClaim, TradeProposal, Position
from .optimizer import solve_optimal_lineup

@dataclass
class AgentParameters:
    """Numerical parameters that dictate an AI's behavior in drafts, lineups, waivers, and trades.
    These can be mutated and tuned by the evolutionary training loop.
    """
    # Draft VORP decay weights (how fast value drops for backup positions)
    vorp_decay_qb: float = 0.5
    vorp_decay_rb: float = 0.4
    vorp_decay_wr: float = 0.4
    vorp_decay_te: float = 0.3
    
    # Draft positional restrictions
    early_rb_limit: int = 99      # max RBs drafted in first 6 rounds (0 for zero-RB, 1 for hero-RB)
    early_qb_limit: int = 99      # max QBs drafted in first 6 rounds (0 for late-round QB)
    early_rb_minimum: int = 0     # force RBs in first 3 rounds (3 for robust RB)
    
    # Draft risk / age parameters
    rookie_boost: float = 1.0     # multiplier for rookie projections
    age_penalty_threshold: int = 30
    age_penalty_factor: float = 0.0 # penalty percentage per year over threshold
    
    # Waiver parameters
    waiver_min_improvement: float = 2.5 # minimum week projection gain to drop/add
    waiver_max_faab_pct: float = 0.05   # maximum FAAB budget percent to bid per player
    
    # Trade parameters
    trade_min_gain: float = 1.0   # minimum starting points gained to accept/propose
    young_player_trade_boost: float = 0.0 # multiplier boost for acquiring young players (<= 24)
    trade_future_discount: float = 0.90   # discount factor for aging/future trade assets
    
    # Lineup / Matchup / Stacking parameters
    matchup_adjustment: float = 0.0 # 0.0 (none) to 1.0 (full weight to opposing team defensive points allowed)
    qb_wr_stack_boost: float = 1.0  # multiplier for drafting WR/TE on the same NFL team as drafted QB
    faab_urgency_factor: float = 1.0 # scaling factor for late-season FAAB aggressiveness



class BaseAgent:
    """The central AI agent class.
    All actions are fully driven by self.params, which allows for genetic algorithm training.
    """
    def __init__(self, team_id: str, settings: LeagueSettings, params: Optional[AgentParameters] = None):
        self.team_id = team_id
        self.settings = settings
        self.params = params or AgentParameters()
        self.defense_factors: Dict[Tuple[str, str], float] = {}

    def get_replacement_levels(self, available_players: List[Player], 
                               projs: Dict[str, float]) -> Dict[str, float]:
        """Estimates replacement level player scores for each position based on league format."""
        teams_count = self.settings.teams_count
        roster = self.settings.roster
        
        qb_count = (roster.qb + roster.superflex * 0.8) * teams_count
        rb_count = (roster.rb + roster.flex * 0.4) * teams_count
        wr_count = (roster.wr + roster.flex * 0.5) * teams_count
        te_count = (roster.te + roster.flex * 0.1) * teams_count
        k_count = teams_count
        dst_count = teams_count
        
        position_targets = {
            "QB": int(qb_count * 1.5),
            "RB": int(rb_count * 1.6),
            "WR": int(wr_count * 1.6),
            "TE": int(te_count * 1.5),
            "K": int(k_count),
            "DST": int(dst_count)
        }
        
        replacement_scores = {}
        for pos, rank in position_targets.items():
            pos_players = [p for p in available_players if p.position.value if hasattr(p.position, "value") and p.position.value == pos or str(p.position) == pos]
            pos_players = sorted(pos_players, key=lambda p: projs.get(p.id, 0.0), reverse=True)
            
            replacement_idx = min(len(pos_players) - 1, rank)
            if replacement_idx >= 0:
                replacement_scores[pos] = projs.get(pos_players[replacement_idx].id, 0.0)
            else:
                defaults = {"QB": 14.0, "RB": 8.0, "WR": 8.0, "TE": 5.0, "K": 6.0, "DST": 6.0}
                replacement_scores[pos] = defaults.get(pos, 5.0)
                
        return replacement_scores

    def get_vorp_scores(self, available_players: List[Player], 
                        projs: Dict[str, float]) -> Dict[str, float]:
        """Calculates VORP score for each player."""
        replacements = self.get_replacement_levels(available_players, projs)
        
        vorp_scores = {}
        for p in available_players:
            pos_str = p.position.value if hasattr(p.position, "value") else str(p.position)
            proj = projs.get(p.id, 0.0)
            rep = replacements.get(pos_str, 5.0)
            vorp_scores[p.id] = proj - rep
            
        return vorp_scores

    def draft_pick(self, draft_state: Any, available_players: List[Player], 
                   projs: Dict[str, float]) -> Player:
        """Determines the draft pick using parameterized VORP adjusted for roster needs."""
        team = [t for t in draft_state.teams if t.id == self.team_id][0]
        vorp = self.get_vorp_scores(available_players, projs)
        
        roster_players = team.roster.all_players()
        qb_drafted = len([p for p in roster_players if p.position == "QB" or (hasattr(p.position, "value") and p.position.value == "QB")])
        rb_drafted = len([p for p in roster_players if p.position == "RB" or (hasattr(p.position, "value") and p.position.value == "RB")])
        wr_drafted = len([p for p in roster_players if p.position == "WR" or (hasattr(p.position, "value") and p.position.value == "WR")])
        te_drafted = len([p for p in roster_players if p.position == "TE" or (hasattr(p.position, "value") and p.position.value == "TE")])
        
        current_round = draft_state.current_round
        
        adjusted_vorp = {}
        for p in available_players:
            p_vorp = vorp.get(p.id, 0.0)
            pos_str = p.position.value if hasattr(p.position, "value") else str(p.position)
            
            # Apply parameterized positional decay factors
            penalty = 1.0
            if pos_str == "QB" and qb_drafted >= self.settings.roster.qb:
                # Use parameterized backup decay for QB
                penalty = self.params.vorp_decay_qb ** (qb_drafted - self.settings.roster.qb + 1)
            elif pos_str == "RB" and rb_drafted >= self.settings.roster.rb:
                penalty = self.params.vorp_decay_rb ** (rb_drafted - self.settings.roster.rb + 1)
            elif pos_str == "WR" and wr_drafted >= self.settings.roster.wr:
                penalty = self.params.vorp_decay_wr ** (wr_drafted - self.settings.roster.wr + 1)
            elif pos_str == "TE" and te_drafted >= self.settings.roster.te:
                penalty = self.params.vorp_decay_te ** (te_drafted - self.settings.roster.te + 1)
                
            # Positional limits & minimums (Zero-RB, Hero-RB, Late-QB, Robust-RB)
            if pos_str == "RB" and current_round <= 6 and rb_drafted >= self.params.early_rb_limit:
                p_vorp = -100.0
            if pos_str == "QB" and current_round <= 6 and qb_drafted >= self.params.early_qb_limit:
                p_vorp = -100.0
            if pos_str == "RB" and current_round <= 3 and rb_drafted < self.params.early_rb_minimum:
                # Force RB selection
                penalty *= 2.5
                
            # Rookie boost
            if p.experience == 0:
                penalty *= self.params.rookie_boost
                
            # Age penalty
            if p.age and p.age > self.params.age_penalty_threshold:
                age_excess = p.age - self.params.age_penalty_threshold
                penalty *= max(0.2, 1.0 - (age_excess * self.params.age_penalty_factor))
                
            # QB-WR/TE Stack Boost
            if self.params.qb_wr_stack_boost != 1.0 and p.nfl_team:
                if pos_str in ["WR", "TE"] and any((rp.position == "QB" or (hasattr(rp.position, "value") and rp.position.value == "QB")) and rp.nfl_team == p.nfl_team for rp in roster_players):
                    penalty *= self.params.qb_wr_stack_boost
                elif pos_str == "QB" and any((rp.position in ["WR", "TE"] or (hasattr(rp.position, "value") and rp.position.value in ["WR", "TE"])) and rp.nfl_team == p.nfl_team for rp in roster_players):
                    penalty *= self.params.qb_wr_stack_boost

            # Force K and DST to late rounds
            total_rounds = self.settings.roster.total_roster_spots()
            if pos_str in ["K", "DST"] and current_round < total_rounds - 2:
                p_vorp = -100.0

            # Live dynamic math calculation variance (+/- 3% live math evaluation)
            live_math_variance = random.gauss(1.0, 0.03)
            adjusted_vorp[p.id] = (p_vorp * penalty) * live_math_variance


        sorted_by_vorp = sorted(available_players, key=lambda p: adjusted_vorp.get(p.id, -999.0), reverse=True)
        return sorted_by_vorp[0]


    def _get_matchup_adjusted_projections(self, roster: Roster, projections: Dict[str, float]) -> Dict[str, float]:
        """Helper to compute projections adjusted for matchup strength if parameter is set."""
        if self.params.matchup_adjustment == 0.0:
            return projections
            
        adjusted_projs = {}
        # Simple lookup fallback based on opponent strength
        # To keep it lightweight and self-contained, we apply a simulated adjustment
        # based on opposing team strength (mocking top defenses vs poor defenses)
        for p in roster.all_players():
            pos_str = p.position.value if hasattr(p.position, "value") else str(p.position)
            proj = projections.get(p.id, 0.0)
            
            # Simulated defensive strength factor
            # In a full simulation we would pull this from historical defense points allowed.
            # Here we apply a standard opponent adjustment scaled by matchup_adjustment weight.
            # We can mock this by looking at player ID hash to keep it stable
            team_hash = hash(p.nfl_team) % 3
            if team_hash == 0: # Hard Matchup
                factor = 1.0 - (0.15 * self.params.matchup_adjustment)
            elif team_hash == 1: # Easy Matchup
                factor = 1.0 + (0.15 * self.params.matchup_adjustment)
            else: # Neutral Matchup
                factor = 1.0
                
            adjusted_projs[p.id] = proj * factor
            
        return adjusted_projs

    def optimize_weekly_lineup(self, roster: Roster, projections: Dict[str, float]) -> Tuple[List[Player], List[Player]]:
        """Sets starting lineup based on standard math optimization (optionally adjusted for matchups)."""
        adj_projs = self._get_matchup_adjusted_projections(roster, projections)
        return solve_optimal_lineup(roster.all_players(), adj_projs, self.settings.roster)

    def get_waiver_claims(self, team: Team, free_agents: List[Player], 
                          projs: Dict[str, float], current_week: int) -> List[WaiverClaim]:
        """Scans the free agent pool and builds waiver claims based on parameters."""
        claims = []
        bench_players = team.roster.bench
        if not bench_players:
            return []
            
        bench_sorted = sorted(bench_players, key=lambda p: projs.get(p.id, 0.0))
        fa_sorted = sorted(free_agents, key=lambda p: projs.get(p.id, 0.0), reverse=True)
        
        claim_index = 0
        for fa in fa_sorted[:5]:
            for worst_bench in bench_sorted:
                fa_proj = projs.get(fa.id, 0.0)
                bench_proj = projs.get(worst_bench.id, 0.0)
                
                # Apply rookie boost to free agent score if relevant
                adj_fa_proj = fa_proj
                if fa.experience == 0:
                    adj_fa_proj *= self.params.rookie_boost
                    
                if adj_fa_proj - bench_proj > self.params.waiver_min_improvement:
                    # Calculate bid using waiver_max_faab_pct and faab_urgency_factor
                    urgency_mult = 1.0 + ((current_week / 14.0) * (self.params.faab_urgency_factor - 1.0))
                    max_bid = int(self.settings.faab_budget * self.params.waiver_max_faab_pct * urgency_mult)
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

    def evaluate_trade_proposal(self, team: Team, proposal: TradeProposal, 
                                projs: Dict[str, float]) -> bool:
        """Evaluates whether to accept a trade proposal based on parameters."""
        current_roster = team.roster.all_players()
        current_starters, _ = solve_optimal_lineup(current_roster, projs, self.settings.roster)
        score_before = sum(projs.get(p.id, 0.0) for p in current_starters)
        
        if proposal.receiver_team_id == self.team_id:
            give = proposal.receiver_sends
            receive = proposal.proposer_sends
        else:
            give = proposal.proposer_sends
            receive = proposal.receiver_sends
            
        # Apply trade boosts (e.g. favoring young players)
        boosted_projs = dict(projs)
        for p in receive:
            if p.age and p.age <= 24:
                boosted_projs[p.id] = projs.get(p.id, 0.0) * (1.0 + self.params.young_player_trade_boost)
                
        hypothetical_roster = [p for p in current_roster if p not in give] + receive
        hypothetical_starters, _ = solve_optimal_lineup(hypothetical_roster, boosted_projs, self.settings.roster)
        score_after = sum(boosted_projs.get(p.id, 0.0) for p in hypothetical_starters)
        
        return (score_after - score_before) >= self.params.trade_min_gain

    def generate_trade_proposals(self, team: Team, all_teams: List[Team], 
                                 projs: Dict[str, float]) -> List[TradeProposal]:
        """Scans other teams' rosters and generates win-win trade proposals.
        All agents can call this, using their own parameterized trade thresholds.
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
                    # Swapping players of different positions to balance rosters
                    if my_player.position == other_player.position:
                        continue
                        
                    # Calculate my score after swap
                    my_hyp_roster = [p for p in my_roster if p.id != my_player.id] + [other_player]
                    my_hyp_starters, _ = solve_optimal_lineup(my_hyp_roster, projs, self.settings.roster)
                    my_score_after = sum(projs.get(p.id, 0.0) for p in my_hyp_starters)
                    my_gain = my_score_after - my_score_before
                    
                    # Calculate other team's score after swap (assuming a standard acceptance margin of 0.5 points)
                    other_hyp_roster = [p for p in other_roster if p.id != other_player.id] + [my_player]
                    other_hyp_starters, _ = solve_optimal_lineup(other_hyp_roster, projs, self.settings.roster)
                    other_score_after = sum(projs.get(p.id, 0.0) for p in other_hyp_starters)
                    other_gain = other_score_after - other_score_before
                    
                    # Propose if both benefit (using my parameterized trade_min_gain)
                    if my_gain >= self.params.trade_min_gain and other_gain >= 0.5:
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
                # Limit to 1 active proposal per week to prevent flooding
                if len(proposals) >= 1:
                    break
                    
        return proposals
