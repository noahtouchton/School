from typing import Dict, Any, Optional
from ..config import ScoringRules
from ..models import Player, Position

def calculate_offensive_points(stats: Dict[str, float], position: str, rules: ScoringRules) -> float:
    """Calculates fantasy points for QBs, RBs, WRs, TEs, and Ks based on scoring rules."""
    pts = 0.0
    
    # Passing
    pts += stats.get("passing_yards", 0.0) * rules.pass_yard
    pts += stats.get("passing_tds", 0.0) * rules.pass_td
    pts += stats.get("interceptions", 0.0) * rules.pass_int
    
    # Rushing
    pts += stats.get("rushing_yards", 0.0) * rules.rush_yard
    pts += stats.get("rushing_tds", 0.0) * rules.rush_td
    
    # Receiving
    pts += stats.get("receiving_yards", 0.0) * rules.rec_yard
    pts += stats.get("receiving_tds", 0.0) * rules.rec_td
    
    receptions = stats.get("receptions", 0.0)
    pts += receptions * rules.rec_reception
    if position == "TE":
        pts += receptions * rules.te_premium_bonus
        
    # Miscellaneous / Fumbles
    pts += stats.get("fumble_lost", 0.0) * rules.fumble_lost
    pts += stats.get("two_point_conversions", 0.0) * rules.rec_2pt
    
    # Kicker stats
    pts += stats.get("fg_made_0_39", 0.0) * 3.0
    pts += stats.get("fg_made_40_49", 0.0) * 4.0
    pts += stats.get("fg_made_50_plus", 0.0) * 5.0
    pts += stats.get("pat_made", 0.0) * 1.0
    pts += stats.get("fg_missed", 0.0) * -1.0
    
    return round(pts, 2)


def calculate_dst_points(points_allowed: float, sacks: int = 0, interceptions: int = 0, 
                         fumble_recoveries: int = 0, safeties: int = 0, touchdowns: int = 0,
                         blocked_kicks: int = 0) -> float:
    """Calculates fantasy points for a defense/special teams unit."""
    pts = 0.0
    
    # Points Allowed scoring tiers
    if points_allowed == 0:
        pts += 10.0
    elif 1 <= points_allowed <= 6:
        pts += 7.0
    elif 7 <= points_allowed <= 13:
        pts += 4.0
    elif 14 <= points_allowed <= 20:
        pts += 1.0
    elif 21 <= points_allowed <= 27:
        pts += 0.0
    elif 28 <= points_allowed <= 34:
        pts += -1.0
    else:  # 35+
        pts += -4.0
        
    # Defensive Plays
    pts += sacks * 1.0
    pts += interceptions * 2.0
    pts += fumble_recoveries * 2.0
    pts += safeties * 2.0
    pts += touchdowns * 6.0
    pts += blocked_kicks * 2.0
    
    return round(pts, 2)


def get_player_fantasy_points(player: Player, stats: Optional[Dict[str, float]], rules: ScoringRules) -> float:
    """Computes a player's fantasy points given their position and stats dictionary."""
    if not stats:
        return 0.0
        
    pos_str = player.position.value if hasattr(player.position, "value") else str(player.position)
    
    if pos_str == "DST":
        # If it's a DST, check if stats dict has direct points allowed.
        # Otherwise, default to a baseline or extract from stats dict
        pts_allowed = stats.get("points_allowed", 21.0)
        sacks = int(stats.get("sacks", 0))
        ints = int(stats.get("interceptions", 0))
        fumbles = int(stats.get("fumble_recoveries", 0))
        safeties = int(stats.get("safeties", 0))
        tds = int(stats.get("defensive_tds", 0))
        return calculate_dst_points(pts_allowed, sacks, ints, fumbles, safeties, tds)
        
    return calculate_offensive_points(stats, pos_str, rules)
