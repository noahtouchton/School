import sqlite3
import json
import os
from typing import List, Dict, Optional, Set, Any
from ..models import Player, Position, PlayerWeeklyStats, PlayerWeeklyProjection

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DB_PATH = os.path.join(DB_DIR, "fantasy_sandbox.db")

def get_connection() -> sqlite3.Connection:
    """Gets an active connection to the SQLite database, creating directories if needed."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes database tables if they do not exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                position TEXT NOT NULL,
                nfl_team TEXT NOT NULL,
                status TEXT NOT NULL,
                injury_status TEXT,
                age INTEGER,
                experience INTEGER
            )
        """)
        
        # Weekly Stats table (stores actual stats dictionary as JSON)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weekly_stats (
                player_id TEXT,
                year INTEGER,
                week INTEGER,
                stats_json TEXT NOT NULL,
                PRIMARY KEY (player_id, year, week),
                FOREIGN KEY (player_id) REFERENCES players (id) ON DELETE CASCADE
            )
        """)
        
        # Weekly Projections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weekly_projections (
                player_id TEXT,
                year INTEGER,
                week INTEGER,
                projected_points REAL NOT NULL,
                projections_json TEXT,
                PRIMARY KEY (player_id, year, week),
                FOREIGN KEY (player_id) REFERENCES players (id) ON DELETE CASCADE
            )
        """)
        
        # Scrape Metadata table (tracks what years/weeks have been imported)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_cache_manifest (
                year INTEGER,
                data_type TEXT, -- 'stats' or 'projections' or 'roster'
                PRIMARY KEY (year, data_type)
            )
        """)
        
        # Trained models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trained_models (
                name TEXT PRIMARY KEY,
                params_json TEXT NOT NULL
            )
        """)
        
        # Seed pre-trained robust AI models out-of-the-box
        try:
            from ..ai.pretrained import PRETRAINED_MODELS
            for m_name, m_params in PRETRAINED_MODELS.items():
                cursor.execute("""
                    INSERT OR IGNORE INTO trained_models (name, params_json)
                    VALUES (?, ?)
                """, (m_name, json.dumps(m_params)))
        except Exception as e:
            print(f"Pre-trained model seeding note: {e}")
            
        # Ensure updated roster assignments
        try:
            cursor.execute("UPDATE players SET nfl_team = 'SF' WHERE LOWER(name) = 'mike evans'")
        except Exception:
            pass

        conn.commit()



# --- Trained Models CRUD operations ---

def save_trained_model(name: str, params: Dict[str, Any]):
    """Saves or updates a trained model's parameter dictionary in SQLite."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO trained_models (name, params_json)
            VALUES (?, ?)
        """, (name, json.dumps(params)))
        conn.commit()

def get_trained_model(name: str) -> Optional[Dict[str, Any]]:
    """Retrieves a trained model's parameter dictionary from SQLite."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT params_json FROM trained_models WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return None
        return json.loads(row["params_json"])

def get_all_trained_models() -> List[str]:
    """Retrieves names of all permanently saved trained models."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM trained_models ORDER BY name")
        rows = cursor.fetchall()
        return [row["name"] for row in rows]

# --- Player CRUD operations ---

def save_players(players: List[Player]):
    """Inserts or replaces players in the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO players (id, name, position, nfl_team, status, injury_status, age, experience)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (p.id, p.name, p.position.value, p.nfl_team, p.status, p.injury_status, p.age, p.experience)
            for p in players
        ])
        conn.commit()

def get_player(player_id: str) -> Optional[Player]:
    """Retrieves a player by their ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM players WHERE id = ?", (player_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return Player(
            id=row["id"],
            name=row["name"],
            position=Position(row["position"]),
            nfl_team=row["nfl_team"],
            status=row["status"],
            injury_status=row["injury_status"],
            age=row["age"],
            experience=row["experience"]
        )

def get_all_players(positions: Optional[List[Position]] = None) -> List[Player]:
    """Retrieves all players, optionally filtered by positions."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if positions:
            pos_strs = [p.value for p in positions]
            placeholders = ",".join("?" for _ in pos_strs)
            cursor.execute(f"SELECT * FROM players WHERE position IN ({placeholders})", pos_strs)
        else:
            cursor.execute("SELECT * FROM players")
        
        rows = cursor.fetchall()
        return [
            Player(
                id=row["id"],
                name=row["name"],
                position=Position(row["position"]),
                nfl_team=row["nfl_team"],
                status=row["status"],
                injury_status=row["injury_status"],
                age=row["age"],
                experience=row["experience"]
            )
            for row in rows
        ]

# --- Weekly Stats CRUD operations ---

def save_weekly_stats(stats_list: List[PlayerWeeklyStats]):
    """Inserts or replaces weekly stats."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO weekly_stats (player_id, year, week, stats_json)
            VALUES (?, ?, ?, ?)
        """, [
            (s.player_id, s.year, s.week, json.dumps(s.stats))
            for s in stats_list
        ])
        conn.commit()

def get_weekly_stats(year: int, week: int, player_id: Optional[str] = None) -> List[PlayerWeeklyStats]:
    """Gets weekly stats for a year/week. If player_id is specified, gets it for that player only."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if player_id:
            cursor.execute("SELECT * FROM weekly_stats WHERE year = ? AND week = ? AND player_id = ?", (year, week, player_id))
        else:
            cursor.execute("SELECT * FROM weekly_stats WHERE year = ? AND week = ?", (year, week))
        
        rows = cursor.fetchall()
        return [
            PlayerWeeklyStats(
                player_id=row["player_id"],
                year=row["year"],
                week=row["week"],
                stats=json.loads(row["stats_json"])
            )
            for row in rows
        ]

# --- Weekly Projections CRUD operations ---

def save_weekly_projections(projections: List[PlayerWeeklyProjection]):
    """Inserts or replaces weekly projections."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO weekly_projections (player_id, year, week, projected_points, projections_json)
            VALUES (?, ?, ?, ?, ?)
        """, [
            (p.player_id, p.year, p.week, p.projected_points, json.dumps(p.stats))
            for p in projections
        ])
        conn.commit()

def get_weekly_projections(year: int, week: int, player_id: Optional[str] = None) -> List[PlayerWeeklyProjection]:
    """Gets projections for a year/week."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if player_id:
            cursor.execute("SELECT * FROM weekly_projections WHERE year = ? AND week = ? AND player_id = ?", (year, week, player_id))
        else:
            cursor.execute("SELECT * FROM weekly_projections WHERE year = ? AND week = ?", (year, week))
        
        rows = cursor.fetchall()
        return [
            PlayerWeeklyProjection(
                player_id=row["player_id"],
                year=row["year"],
                week=row["week"],
                projected_points=row["projected_points"],
                stats=json.loads(row["projections_json"]) if row["projections_json"] else {}
            )
            for row in rows
        ]

# --- Manifest Operations (Cache Tracking) ---

def is_year_cached(year: int, data_type: str) -> bool:
    """Checks if a specific year's data is already successfully cached."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM data_cache_manifest WHERE year = ? AND data_type = ?", (year, data_type))
        return cursor.fetchone() is not None

def mark_year_cached(year: int, data_type: str):
    """Marks a specific year's data as cached."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO data_cache_manifest (year, data_type) VALUES (?, ?)", (year, data_type))
        conn.commit()

# Automatically initialize database when module is imported
init_db()
