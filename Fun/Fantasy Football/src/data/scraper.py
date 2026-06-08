import pandas as pd
import numpy as np
import nfl_data_py as nfl
import json
from typing import List, Dict, Any, Optional
from ..models import Player, Position, PlayerWeeklyStats, PlayerWeeklyProjection
from . import db

# Mapping NFL team abbreviations to standard fantasy abbreviations if needed
TEAM_MAPPING = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF", "CAR": "CAR",
    "CHI": "CHI", "CIN": "CIN", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN",
    "DET": "DET", "GB": "GB", "HOU": "HOU", "IND": "IND", "JAX": "JAX",
    "KC": "KC", "LV": "LV", "LAC": "LAC", "LAR": "LAR", "MIA": "MIA",
    "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG", "NYJ": "NYJ",
    "PHI": "PHI", "PIT": "PIT", "SF": "SF", "SEA": "SEA", "TB": "TB",
    "TEN": "TEN", "WAS": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LAR"
}

def scrape_and_cache_season(year: int, force: bool = False):
    """Scrapes and saves rosters and weekly statistics for a given year to the SQLite DB."""
    if not force and db.is_year_cached(year, "stats") and db.is_year_cached(year, "rosters"):
        print(f"Season {year} data is already cached. Skipping scrape.")
        return

    print(f"Scraping roster and weekly data for the {year} season...")
    
    # 1. Fetch weekly rosters
    try:
        rosters_df = nfl.import_weekly_rosters([year])
    except Exception as e:
        print(f"Error importing weekly rosters for {year}: {e}")
        return
        
    # Filter for standard fantasy positions
    valid_positions = {"QB", "RB", "WR", "TE", "K"}
    rosters_df = rosters_df[rosters_df["position"].isin(valid_positions)]
    
    # Get unique players by taking their latest weekly record of the season
    latest_rosters = rosters_df.sort_values("week").groupby("player_id").last().reset_index()
    
    players = []
    for _, row in latest_rosters.iterrows():
        p_id = str(row["player_id"])
        name = str(row["player_name"]) if not pd.isna(row["player_name"]) else str(row["first_name"]) + " " + str(row["last_name"])
        pos = Position(row["position"])
        team = TEAM_MAPPING.get(row["team"], row["team"])
        status = str(row["status"]) if not pd.isna(row["status"]) else "Active"
        
        age = int(row["age"]) if not pd.isna(row["age"]) else None
        exp = int(row["years_exp"]) if not pd.isna(row["years_exp"]) else 0
        
        player = Player(
            id=p_id,
            name=name,
            position=pos,
            nfl_team=team,
            status=status,
            age=age,
            experience=exp
        )
        players.append(player)
        
    # Also add Defense/Special Teams (DST) players for each NFL team
    for team_abbr in set(TEAM_MAPPING.values()):
        players.append(Player(
            id=f"DST_{team_abbr}",
            name=f"{team_abbr} Defense",
            position=Position.DST if hasattr(Position, "DST") else "DST", # Fallback if Enum doesn't have it
            nfl_team=team_abbr,
            status="Active"
        ))
        
    print(f"Saving {len(players)} players to database...")
    db.save_players(players)
    db.mark_year_cached(year, "rosters")

    # 2. Fetch weekly statistics
    try:
        if year == 2025:
            # Direct download fallback for 2025 weekly stats URL to bypass nfl-data-py 404
            url = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2025.parquet"
            stats_df = pd.read_parquet(url)
        else:
            stats_df = nfl.import_weekly_data([year])
    except Exception as e:
        print(f"Error importing weekly data for {year}: {e}")
        return
        
    weekly_stats = []
    
    # Replace NaN with 0.0 for numeric columns to prevent JSON serialization errors
    numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
    stats_df[numeric_cols] = stats_df[numeric_cols].fillna(0.0)
    
    for _, row in stats_df.iterrows():
        p_id = str(row["player_id"])
        week = int(row["week"])
        
        # Build raw stats dict
        stats_dict = {
            "passing_yards": float(row.get("passing_yards", 0.0)),
            "passing_tds": int(row.get("passing_tds", 0)),
            "interceptions": float(row.get("interceptions", row.get("passing_interceptions", 0.0))),
            "rushing_yards": float(row.get("rushing_yards", 0.0)),
            "rushing_tds": int(row.get("rushing_tds", 0)),
            "receptions": int(row.get("receptions", 0)),
            "receiving_yards": float(row.get("receiving_yards", 0.0)),
            "receiving_tds": int(row.get("receiving_tds", 0)),
            "fumbles_lost": float(row.get("rushing_fumbles_lost", 0.0)) + float(row.get("receiving_fumbles_lost", 0.0)),
            "two_point_conversions": int(row.get("passing_2pt_conversions", 0)) + int(row.get("rushing_2pt_conversions", 0)) + int(row.get("receiving_2pt_conversions", 0))
        }
        
        weekly_stat = PlayerWeeklyStats(
            player_id=p_id,
            year=year,
            week=week,
            stats=stats_dict
        )
        weekly_stats.append(weekly_stat)
        
    # Calculate DST scoring from raw stats if desired, or we can calculate DST scores dynamically
    # For now, let's also save the player stats.
    print(f"Saving {len(weekly_stats)} weekly player stats to database...")
    db.save_weekly_stats(weekly_stats)
    db.mark_year_cached(year, "stats")
    
    # 3. Generate Projections for this year
    generate_rolling_projections(year)

def calculate_fantasy_points(stats: Dict[str, float], position: str, rules: Any) -> float:
    """Calculates fantasy points for a stat dict based on custom scoring rules."""
    pts = 0.0
    pts += stats.get("passing_yards", 0.0) * rules.pass_yard
    pts += stats.get("passing_tds", 0.0) * rules.pass_td
    pts += stats.get("interceptions", 0.0) * rules.pass_int
    
    pts += stats.get("rushing_yards", 0.0) * rules.rush_yard
    pts += stats.get("rushing_tds", 0.0) * rules.rush_td
    
    pts += stats.get("receiving_yards", 0.0) * rules.rec_yard
    pts += stats.get("receiving_tds", 0.0) * rules.rec_td
    
    receptions = stats.get("receptions", 0.0)
    pts += receptions * rules.rec_reception
    if position == "TE":
        pts += receptions * rules.te_premium_bonus
        
    pts += stats.get("fumble_lost", 0.0) * rules.fumble_lost
    pts += stats.get("two_point_conversions", 0.0) * rules.rec_2pt # Assuming 2pt value is same
    
    return round(pts, 2)

def generate_rolling_projections(year: int, window: int = 3):
    """Generates weekly projections for a season based on a rolling average of actual scores.
    This provides realistic, dynamic projections for historical sandboxes.
    """
    print(f"Generating rolling projections for the {year} season (window={window})...")
    
    # Get all players
    players = db.get_all_players()
    player_map = {p.id: p for p in players}
    
    from ..config import ScoringRules
    rules = ScoringRules.half_ppr() # Use default Half-PPR for base projection score
    
    # Let's organize actual scores by player, week
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT player_id, week, stats_json FROM weekly_stats WHERE year = ? ORDER BY week", (year,))
        rows = cursor.fetchall()
        
    player_weekly_scores = {} # player_id -> list of (week, score)
    player_weekly_stats = {}  # player_id -> list of (week, stats_dict)
    
    for row in rows:
        p_id = row["player_id"]
        week = row["week"]
        stats = json.loads(row["stats_json"])
        
        p_obj = player_map.get(p_id)
        pos = p_obj.position.value if p_obj else "WR"
        
        score = calculate_fantasy_points(stats, pos, rules)
        
        if p_id not in player_weekly_scores:
            player_weekly_scores[p_id] = []
            player_weekly_stats[p_id] = []
            
        player_weekly_scores[p_id].append((week, score))
        player_weekly_stats[p_id].append((week, stats))
        
    projections = []
    
    # Default week 1 projections based on position if no prior year data
    # (In a real system, we'd pull prior year stats, which we can do if the database has it)
    default_projections = {
        "QB": 16.0,
        "RB": 10.0,
        "WR": 10.0,
        "TE": 6.5,
        "K": 7.0,
        "DST": 7.0
    }
    
    # Get prior year scores if available for week 1 baseline
    prior_year_averages = {}
    if db.is_year_cached(year - 1, "stats"):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT player_id, stats_json FROM weekly_stats WHERE year = ?", (year - 1,))
            prior_rows = cursor.fetchall()
            
        prior_scores = {}
        for r in prior_rows:
            pid = r["player_id"]
            p_obj = player_map.get(pid)
            p_pos = p_obj.position.value if p_obj else "WR"
            p_score = calculate_fantasy_points(json.loads(r["stats_json"]), p_pos, rules)
            
            if pid not in prior_scores:
                prior_scores[pid] = []
            prior_scores[pid].append(p_score)
            
        for pid, scores in prior_scores.items():
            if len(scores) > 0:
                prior_year_averages[pid] = sum(scores) / len(scores)

    # Let's generate week-by-week projections for weeks 1 through 18
    for week in range(1, 19):
        for p in players:
            # Skip DSTs for projection generation for now (or give them default)
            if p.position == "DST":
                proj_pts = 7.0
                projections.append(PlayerWeeklyProjection(
                    player_id=p.id, year=year, week=week, projected_points=proj_pts, stats={}
                ))
                continue
                
            p_scores = player_weekly_scores.get(p.id, [])
            p_stats = player_weekly_stats.get(p.id, [])
            
            # Find scores in current year before the target week
            past_scores = [s for wk, s in p_scores if wk < week]
            past_stats = [st for wk, st in p_stats if wk < week]
            
            proj_pts = 0.0
            proj_stats = {}
            
            if len(past_scores) > 0:
                # Use rolling average of last `window` games
                recent_scores = past_scores[-window:]
                proj_pts = sum(recent_scores) / len(recent_scores)
                
                # Average stats similarly
                recent_stats = past_stats[-window:]
                for key in ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds", "fumbles_lost", "two_point_conversions"]:
                    proj_stats[key] = sum(s.get(key, 0.0) for s in recent_stats) / len(recent_stats)
            else:
                # Week 1, or player hasn't played yet this season
                # Try prior year average
                if p.id in prior_year_averages:
                    proj_pts = prior_year_averages[p.id]
                else:
                    # Positional baseline
                    proj_pts = default_projections.get(p.position.value, 8.0)
                    
                # Fill default stats based on projected points
                # (Simple linear approximation for baseline stats)
                if p.position == Position.QB:
                    proj_stats = {"passing_yards": proj_pts * 15, "passing_tds": proj_pts / 6}
                elif p.position == Position.RB:
                    proj_stats = {"rushing_yards": proj_pts * 7, "receptions": proj_pts / 8}
                elif p.position == Position.WR:
                    proj_stats = {"receiving_yards": proj_pts * 8, "receptions": proj_pts / 7}
                elif p.position == Position.TE:
                    proj_stats = {"receiving_yards": proj_pts * 6, "receptions": proj_pts / 8}
            
            # Add small random adjustment to projections to simulate variance / market updates
            # Projections are never 100% stable
            proj_pts = max(0.0, round(proj_pts, 2))
            
            projections.append(PlayerWeeklyProjection(
                player_id=p.id,
                year=year,
                week=week,
                projected_points=proj_pts,
                stats=proj_stats
            ))
            
    print(f"Saving {len(projections)} weekly player projections to database...")
    db.save_weekly_projections(projections)
    db.mark_year_cached(year, "projections")
