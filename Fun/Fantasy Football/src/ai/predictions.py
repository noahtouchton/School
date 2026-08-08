import json
from typing import List, Dict, Tuple, Optional, Any
from ..models import Player, Position, PlayerWeeklyProjection
from ..data import db

# Updated NFL Team Assignments & News Notes for key marquee players
UPDATED_PLAYER_NEWS: Dict[str, Dict[str, str]] = {
    "mike evans": {
        "nfl_team": "SF",
        "news": "🌉 Joined San Francisco 49ers as high-upside WR threat in Kyle Shanahan's explosive offense.",
        "forecast": "Strong WR1/WR2 boundary volume based on endzone target share and 49ers passing scheme."
    },

    "saquon barkley": {
        "nfl_team": "PHI",
        "news": "🦅 Featured bellcow RB behind elite Eagles offensive line. High volume goal-line & receiving usage.",
        "forecast": "Top 3 overall fantasy RB projections heading into the season."
    },
    "derrick henry": {
        "nfl_team": "BAL",
        "news": "🐦 Ravens lead-back role in high-powered rushing offense alongside Lamar Jackson.",
        "forecast": "Elite touchdown upside in heavy positive game scripts."
    },
    "stefon diggs": {
        "nfl_team": "HOU",
        "news": "🚀 Primary target option in C.J. Stroud's explosive Houston passing attack.",
        "forecast": "High PPR reception floor with elite slot/outside versatility."
    },
    "christian mccaffrey": {
        "nfl_team": "SF",
        "news": "⚡ Fully healthy 49ers focal point in Kyle Shanahan's offense.",
        "forecast": "Overall #1 fantasy scoring projection per game across all formats."
    },
    "justin jefferson": {
        "nfl_team": "MIN",
        "news": "🎯 Alpha WR1 with dominant 30%+ team target share.",
        "forecast": "Elite receiving yardage projections with high weekly floor."
    },
    "c.j. stroud": {
        "nfl_team": "HOU",
        "news": "📈 Year 3 leap expected in high-volume passing offense with upgraded receiving corps.",
        "forecast": "Top 5 fantasy QB projection with high passing yardage ceiling."
    }
}

class AINewsPredictionEngine:
    """Generates AI player news notes, updated team assignments, and 2026 predictions
    weighted heavily on previous season performance.
    """
    def __init__(self):
        pass

    def apply_updated_rosters_and_news(self, players: List[Player]) -> List[Player]:
        """Applies latest NFL team assignments and news annotations to player objects."""
        for p in players:
            p_name_lower = p.name.lower()
            if p_name_lower in UPDATED_PLAYER_NEWS:
                info = UPDATED_PLAYER_NEWS[p_name_lower]
                p.nfl_team = info["nfl_team"]
        return players

    def get_player_news(self, player_name: str) -> Optional[Dict[str, str]]:
        """Retrieves news note and forecast for a player if available."""
        return UPDATED_PLAYER_NEWS.get(player_name.lower())

    def get_all_news_highlights(() -> List[Dict[str, str]]:
        pass

    def get_all_news_highlights(self) -> List[Dict[str, str]]:
        """Returns a list of all active marquee news highlights for UI display."""
        highlights = []
        for p_name, info in UPDATED_PLAYER_NEWS.items():
            highlights.append({
                "player_name": p_name.title(),
                "team": info["nfl_team"],
                "news": info["news"],
                "forecast": info["forecast"]
            })
        return highlights

    def compute_weighted_projections(self, year: int, week: int) -> Dict[str, float]:
        """Calculates projections giving 80% weight to previous season stats (e.g. 2025) and 20% to historical average."""
        db.init_db()
        
        # Pull 2025 weekly projections/stats as primary baseline
        prev_year = year - 1 if db.is_year_cached(year - 1, "projections") else year
        base_projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(prev_year, week)}
        curr_projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(year, week)}
        
        combined_projs = {}
        all_p_ids = set(base_projs.keys()) | set(curr_projs.keys())
        
        for p_id in all_p_ids:
            p_prev = base_projs.get(p_id, curr_projs.get(p_id, 8.0))
            p_curr = curr_projs.get(p_id, p_prev)
            
            # 80% previous season performance, 20% current baseline
            weighted_score = (0.80 * p_prev) + (0.20 * p_curr)
            combined_projs[p_id] = round(weighted_score, 2)
            
        return combined_projs
