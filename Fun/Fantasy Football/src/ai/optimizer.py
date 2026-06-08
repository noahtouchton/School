import pulp
from typing import List, Dict, Tuple, Optional
from ..models import Player, Roster, Position
from ..config import RosterSettings

def solve_optimal_lineup_greedy(players: List[Player], projections: Dict[str, float],
                               settings: RosterSettings) -> Tuple[List[Player], List[Player]]:
    """Greedy fallback to select the optimal starting lineup if PuLP solver fails."""
    starters = []
    remaining = list(players)
    
    # Sort remaining players by projection descending
    remaining = sorted(remaining, key=lambda p: projections.get(p.id, 0.0), reverse=True)
    
    # Helper to select next best eligible player for a slot
    def fill_slot(pos_filter, count):
        filled = 0
        i = 0
        while filled < count and i < len(remaining):
            p = remaining[i]
            pos_str = p.position.value if hasattr(p.position, "value") else str(p.position)
            if pos_filter is None or pos_str == pos_filter or (isinstance(pos_filter, list) and pos_str in pos_filter):
                starters.append(p)
                remaining.remove(p)
                filled += 1
            else:
                i += 1
                
    # 1. Fill primary spots
    fill_slot("QB", settings.qb)
    fill_slot("RB", settings.rb)
    fill_slot("WR", settings.wr)
    fill_slot("TE", settings.te)
    
    # 2. Fill FLEX spots (RB/WR/TE)
    fill_slot(["RB", "WR", "TE"], settings.flex)
    
    # 3. Fill SUPERFLEX spots (QB/RB/WR/TE)
    fill_slot(["QB", "RB", "WR", "TE"], settings.superflex)
    
    bench = list(remaining)
    return starters, bench

def solve_optimal_lineup(players: List[Player], projections: Dict[str, float], 
                         settings: RosterSettings) -> Tuple[List[Player], List[Player]]:
    """Uses Mixed-Integer Linear Programming (PuLP) to select the optimal starting lineup
    maximizing projected points under positional and flex/superflex constraints.
    Falls back to a greedy solver if PuLP encounters execution errors.
    """
    if not players:
        return [], []

    try:
        # Filter out inactive/empty players
        eligible_players = [p for p in players]
        
        # 1. Create optimization problem
        prob = pulp.LpProblem("Optimal_Lineup", pulp.LpMaximize)
        
        # Define starter slot types
        slots = ["QB", "RB", "WR", "TE", "FLEX", "SUPERFLEX"]
        starter_counts = {
            "QB": settings.qb,
            "RB": settings.rb,
            "WR": settings.wr,
            "TE": settings.te,
            "FLEX": settings.flex,
            "SUPERFLEX": settings.superflex
        }
        
        # 2. Define Decision Variables
        y = {}
        for p in eligible_players:
            pos = p.position.value if hasattr(p.position, "value") else str(p.position)
            for slot in slots:
                # Check eligibility for this slot
                is_eligible = False
                if slot == "QB" and pos == "QB":
                    is_eligible = True
                elif slot == "RB" and pos == "RB":
                    is_eligible = True
                elif slot == "WR" and pos == "WR":
                    is_eligible = True
                elif slot == "TE" and pos == "TE":
                    is_eligible = True
                elif slot == "FLEX" and pos in ["RB", "WR", "TE"]:
                    is_eligible = True
                elif slot == "SUPERFLEX" and pos in ["QB", "RB", "WR", "TE"]:
                    is_eligible = True
                    
                if is_eligible:
                    y[(p.id, slot)] = pulp.LpVariable(f"start_{p.id}_{slot}", cat=pulp.LpBinary)
                    
        # 3. Objective Function: Maximize projected points
        prob += pulp.lpSum(
            y[(p.id, slot)] * projections.get(p.id, 0.0)
            for (p_id, slot), var in y.items()
            for p in eligible_players if p.id == p_id
        )
        
        # 4. Constraints
        for p in eligible_players:
            player_slots = [y[(p.id, slot)] for slot in slots if (p.id, slot) in y]
            if player_slots:
                prob += pulp.lpSum(player_slots) <= 1
                
        for slot in slots:
            slot_players = [y[(p.id, slot)] for p in eligible_players if (p.id, slot) in y]
            capacity = starter_counts.get(slot, 0)
            prob += pulp.lpSum(slot_players) <= capacity
            prob += pulp.lpSum(slot_players) >= min(capacity, len([p for p in eligible_players if (p.id, slot) in y]))
    
        # 5. Solve
        # Use GLPK or CBC default solver silently
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # 6. Extract Results
        starters = []
        starter_ids = set()
        
        for p in eligible_players:
            started = False
            for slot in slots:
                if (p.id, slot) in y and pulp.value(y[(p.id, slot)]) == 1:
                    started = True
                    break
            if started:
                starters.append(p)
                starter_ids.add(p.id)
                
        bench = [p for p in players if p.id not in starter_ids]
        
        # If solver returned an invalid result or failed to find values
        if not starters and any(projections.get(p.id, 0.0) > 0 for p in players):
            return solve_optimal_lineup_greedy(players, projections, settings)
            
        return starters, bench
    except Exception as e:
        # Fallback to greedy on any solver exception
        return solve_optimal_lineup_greedy(players, projections, settings)
