import random
from typing import List, Dict, Tuple, Optional, Any
from ..models import Player, Position, Roster, Team
from ..config import LeagueSettings, ScoringRules, RosterSettings
from ..data import db

try:
    from espn_api.football import League as ESPNLeague
    HAS_ESPN_API = True
except ImportError:
    HAS_ESPN_API = False

POSITION_MAP = {
    "QB": Position.QB,
    "RB": Position.RB,
    "WR": Position.WR,
    "TE": Position.TE,
    "K": Position.K,
    "D/ST": Position.DST,
    "DST": Position.DST,
}

class ESPNClient:
    """Handles communication with ESPN Fantasy Football leagues (both live and mock/demo)."""
    def __init__(self, league_id: int, year: int = 2025, espn_s2: str = "", swid: str = ""):
        self.league_id = league_id
        self.year = year
        self.espn_s2 = espn_s2
        self.swid = swid
        self.espn_league: Optional[Any] = None
        self.is_mock: bool = False

    def connect(self) -> bool:
        """Attempts to connect to the ESPN league. Returns True if connected or fallback mock ready."""
        if self.league_id == 0 or not HAS_ESPN_API:
            self.is_mock = True
            return True
            
        try:
            kwargs = {}
            if self.espn_s2 and self.swid:
                kwargs["espn_s2"] = self.espn_s2
                kwargs["swid"] = self.swid
                
            self.espn_league = ESPNLeague(league_id=self.league_id, year=self.year, **kwargs)
            self.is_mock = False
            return True
        except Exception as e:
            print(f"ESPN API connection failed: {e}. Falling back to mock league mode.")
            self.is_mock = True
            return True

    def get_league_settings(self) -> LeagueSettings:
        """Extracts scoring and roster settings from ESPN or defaults."""
        if not self.is_mock and self.espn_league:
            try:
                # Estimate settings from ESPN league
                teams_cnt = len(self.espn_league.teams)
                return LeagueSettings(
                    name=getattr(self.espn_league, "name", f"ESPN League {self.league_id}"),
                    teams_count=teams_cnt,
                    scoring=ScoringRules.half_ppr(),
                    roster=RosterSettings(qb=1, rb=2, wr=2, te=1, flex=2, bench=6),
                    faab_budget=100
                )
            except Exception:
                pass
                
        return LeagueSettings(
            name="ESPN Fantasy League",
            teams_count=10,
            scoring=ScoringRules.half_ppr(),
            roster=RosterSettings(qb=1, rb=2, wr=2, te=1, flex=2, bench=6),
            faab_budget=100
        )

    def get_teams(self) -> List[Team]:
        """Returns ESPN league teams converted to internal Team data structures."""
        return self.get_teams_internal()

    def get_teams_internal(self) -> List[Team]:

        """Converts ESPN league teams into internal Team data structures."""
        db.init_db()
        all_players = db.get_all_players()
        player_map = {p.name.lower(): p for p in all_players}

        teams = []
        if not self.is_mock and self.espn_league:
            try:
                for espn_team in self.espn_league.teams:
                    roster = Roster()
                    for espn_player in espn_team.roster:
                        p_name = espn_player.name.lower()
                        match = player_map.get(p_name)
                        if not match:
                            pos_str = getattr(espn_player, "position", "WR")
                            pos = POSITION_MAP.get(pos_str, Position.WR)
                            match = Player(
                                id=f"espn_{espn_player.playerId}",
                                name=espn_player.name,
                                position=pos,
                                nfl_team=getattr(espn_player, "proTeam", "FA"),
                                status=getattr(espn_player, "status", "Active")
                            )
                            
                        # Check slot position
                        slot = getattr(espn_player, "lineupSlot", "BE")
                        if slot in ["BE", "BENCH", "IR"]:
                            roster.bench.append(match)
                        else:
                            roster.starters.append(match)

                    team = Team(
                        id=str(espn_team.team_id),
                        name=espn_team.team_name,
                        owner_persona="espn_owner",
                        roster=roster,
                        faab_balance=getattr(espn_team, "faab", 100),
                        wins=getattr(espn_team, "wins", 0),
                        losses=getattr(espn_team, "losses", 0),
                        points_for=getattr(espn_team, "points_for", 0.0)
                    )
                    teams.append(team)
                return teams
            except Exception as e:
                print(f"Error parsing ESPN teams: {e}. Falling back to mock data.")

        # Fallback Mock Teams generator for seamless offline demoing
        sample_names = ["My Franchise", "Gridiron Gladiators", "Blitz Brigade", "Touchdown Titans",
                        "Endzone Experts", "Redzone Raiders", "Field Goal Freaks", "Pigskin Pros",
                        "Huddle Heroes", "Punt Pirates"]
        
        # Shuffle players into 10 teams
        shuffled_p = list(all_players)
        random.seed(42) # Deterministic mock setup
        random.shuffle(shuffled_p)
        
        chunk_size = 15
        for i in range(min(10, len(sample_names))):
            t_players = shuffled_p[i*chunk_size : (i+1)*chunk_size]
            starters = t_players[:9]
            bench = t_players[9:]
            
            teams.append(Team(
                id=f"team_{i+1}",
                name=sample_names[i],
                owner_persona="human" if i == 0 else "espn_opponent",
                roster=Roster(starters=starters, bench=bench),
                faab_balance=85,
                wins=4,
                losses=2,
                points_for=650.0
            ))
            
        return teams

    def get_free_agents_internal(self, week: int = 1) -> List[Player]:
        """Fetches free agents available in the ESPN league."""
        db.init_db()
        all_players = db.get_all_players()
        
        if not self.is_mock and self.espn_league:
            try:
                espn_fas = self.espn_league.free_agents(week=week, size=50)
                fa_players = []
                player_map = {p.name.lower(): p for p in all_players}
                for fa in espn_fas:
                    match = player_map.get(fa.name.lower())
                    if match:
                        fa_players.append(match)
                    else:
                        pos_str = getattr(fa, "position", "WR")
                        pos = POSITION_MAP.get(pos_str, Position.WR)
                        fa_players.append(Player(
                            id=f"espn_fa_{fa.playerId}",
                            name=fa.name,
                            position=pos,
                            nfl_team=getattr(fa, "proTeam", "FA"),
                            status="Active"
                        ))
                return fa_players
            except Exception as e:
                print(f"Error fetching ESPN free agents: {e}")

        # Mock fallback: return top unassigned players
        teams = self.get_teams_internal()
        assigned_ids = {p.id for t in teams for p in t.roster.all_players()}
        return [p for p in all_players if p.id not in assigned_ids][:50]
