import random
import pandas as pd
import nfl_data_py as nfl
from typing import List, Dict, Tuple, Set, Any, Optional
from ..config import LeagueSettings, RosterSettings
from ..models import Player, Team, Roster, Matchup, DraftState, WaiverClaim, TradeProposal
from ..data import db
from .scoring import get_player_fantasy_points

class LeagueSandbox:
    """The central engine that manages a fake league's lifecycle:
    Drafting, scheduling, weekly lineups, waiver claims, trades, and matchup scoring.
    """
    def __init__(self, settings: LeagueSettings, year: int):
        self.settings = settings
        self.year = year
        self.current_week = 1
        self.teams: Dict[str, Team] = {}
        self.draft_state: Optional[DraftState] = None
        self.schedule: Dict[int, List[Matchup]] = {} # week -> list of Matchups
        self.waiver_priority: List[str] = [] # Team IDs ordered by priority (worst record first)
        self.transaction_history: List[Dict[str, Any]] = []

    def add_team(self, team: Team):
        self.teams[team.id] = team

    def initialize_league(self, teams: List[Team]):
        """Sets up the league with teams, draft order, and schedules."""
        if len(teams) != self.settings.teams_count:
            raise ValueError(f"League requires exactly {self.settings.teams_count} teams, but got {len(teams)}.")
            
        self.teams = {t.id: t for t in teams}
        
        # Initialize waiver priority (initially random)
        self.waiver_priority = [t.id for t in teams]
        random.shuffle(self.waiver_priority)
        
        # Generate schedule
        self.generate_schedule()

    def generate_schedule(self):
        """Generates a standard round-robin schedule for a 14-week fantasy season."""
        team_ids = list(self.teams.keys())
        n = len(team_ids)
        
        # Ensure even number of teams
        if n % 2 != 0:
            raise ValueError("League must have an even number of teams to schedule matchups.")
            
        temp_teams = list(team_ids)
        num_weeks = 14  # Standard fantasy regular season
        
        for week in range(1, num_weeks + 1):
            self.schedule[week] = []
            for i in range(n // 2):
                team_a = temp_teams[i]
                team_b = temp_teams[n - 1 - i]
                self.schedule[week].append(Matchup(
                    week=week,
                    team_a_id=team_a,
                    team_b_id=team_b
                ))
            # Rotate list (fixing first index) for the next week's pairings
            temp_teams = [temp_teams[0]] + [temp_teams[-1]] + temp_teams[1:-1]

    def start_draft(self) -> DraftState:
        """Starts a snake draft, generating the draft order."""
        team_ids = list(self.teams.keys())
        # Randomize first round order
        random.shuffle(team_ids)
        
        # Number of rounds equals total roster spots
        rounds = self.settings.roster.total_roster_spots()
        
        self.draft_state = DraftState(
            year=self.year,
            rounds=rounds,
            teams=list(self.teams.values()),
            draft_order=team_ids
        )
        return self.draft_state

    def execute_draft_pick(self, player: Player):
        """Executes the current draft pick on behalf of the team whose turn it is."""
        if not self.draft_state:
            raise ValueError("Draft has not been started.")
            
        team_id = self.draft_state.get_current_team_id()
        team = self.teams[team_id]
        
        # Add to team roster
        self.draft_state.draft_player(team_id, player)
        team.roster.add_player(player)
        
        self.transaction_history.append({
            "type": "DRAFT",
            "week": 0,
            "team_name": team.name,
            "player_name": player.name,
            "position": player.position.value if hasattr(player.position, "value") else player.position,
            "details": f"Drafted in Round {self.draft_state.current_round} (Pick {len(self.draft_state.picks)})"
        })

    def auto_draft_fill(self):
        """Helper to run the rest of the draft using best available projections (auto-draft).
        Useful for testing before human or AI logic is executed.
        """
        if not self.draft_state:
            raise ValueError("Draft has not been started.")
            
        # Get all players sorted by base projection or ADP
        # Let's query players and their average stats
        players = db.get_all_players()
        # Sort by positional importance and filter already drafted
        # QB, RB, WR, TE priority
        pos_rank = {"QB": 1, "RB": 2, "WR": 3, "TE": 4, "K": 5, "DST": 6}
        
        while len(self.draft_state.picks) < self.draft_state.rounds * len(self.teams):
            current_team_id = self.draft_state.get_current_team_id()
            current_team = self.teams[current_team_id]
            
            # Find a valid position the team still needs, or draft best available flex
            undrafted = [p for p in players if p.id not in self.draft_state.drafted_player_ids]
            if not undrafted:
                break
                
            # Basic auto-draft selects highest age/exp or random for fallback
            selected_player = random.choice(undrafted[:20]) # pick one of top 20 random to simulate variety
            self.execute_draft_pick(selected_player)

    def set_lineup(self, team_id: str, starters: List[Player], bench: List[Player], ir: List[Player]):
        """Updates a team's lineup configuration for the week."""
        team = self.teams.get(team_id)
        if not team:
            raise ValueError(f"Team {team_id} not found.")
            
        # In a real league, we validate roster slots
        team.roster.starters = starters
        team.roster.bench = bench
        team.roster.ir = ir

    def process_waiver_claims(self, claims: List[WaiverClaim]):
        """Resolves waiver claims for the current week.
        Supports both FAAB (highest bid wins) and Waiver Priority.
        """
        # Sort claims by bid amount (highest first), then priority order, then waiver priority
        # Let's map waiver priority to an index rank (lower is better)
        priority_rank = {team_id: idx for idx, team_id in enumerate(self.waiver_priority)}
        
        # Group claims by player to add
        claims_by_player: Dict[str, List[WaiverClaim]] = {}
        for claim in claims:
            p_id = claim.player_to_add.id
            if p_id not in claims_by_player:
                claims_by_player[p_id] = []
            claims_by_player[p_id].append(claim)
            
        # Process claims player-by-player
        # If FAAB league:
        claims_sorted_by_bid = sorted(
            claims,
            key=lambda c: (c.bid_amount, -c.priority_order, -priority_rank[c.team_id]),
            reverse=True
        )
        
        processed_players: Set[str] = set()
        
        for claim in claims_sorted_by_bid:
            player_add = claim.player_to_add
            team = self.teams[claim.team_id]
            
            if player_add.id in processed_players:
                # Player already won by a higher bid/priority
                continue
                
            # Verify team can afford bid
            if claim.bid_amount > team.faab_balance:
                continue
                
            # Execute Transaction
            try:
                # Drop player if required
                if claim.player_to_drop:
                    team.roster.remove_player(claim.player_to_drop)
                    
                # Add player
                team.roster.add_player(player_add)
                team.faab_balance -= claim.bid_amount
                processed_players.add(player_add.id)
                
                # Update waiver priority (move winning team to back of priority)
                self.waiver_priority.remove(claim.team_id)
                self.waiver_priority.append(claim.team_id)
                
                self.transaction_history.append({
                    "type": "WAIVER",
                    "week": self.current_week,
                    "team_name": team.name,
                    "player_name": player_add.name,
                    "position": player_add.position.value if hasattr(player_add.position, "value") else player_add.position,
                    "details": f"Added for ${claim.bid_amount}. Dropped {claim.player_to_drop.name if claim.player_to_drop else 'None'}."
                })
            except Exception as e:
                # E.g. player drop failed
                print(f"Failed waiver claim execution: {e}")

    def execute_trade(self, proposal: TradeProposal) -> bool:
        """Executes a finalized trade proposal between two franchises."""
        team_prop = self.teams[proposal.proposer_team_id]
        team_recv = self.teams[proposal.receiver_team_id]
        
        # Verify proposer has all proposed players
        for p in proposal.proposer_sends:
            if p not in team_prop.roster.all_players():
                print(f"Trade failed: proposer {team_prop.name} no longer has player {p.name}")
                return False
                
        # Verify receiver has all receiver sends players
        for p in proposal.receiver_sends:
            if p not in team_recv.roster.all_players():
                print(f"Trade failed: receiver {team_recv.name} no longer has player {p.name}")
                return False
                
        # Remove players from proposer and add to receiver
        for p in proposal.proposer_sends:
            team_prop.roster.remove_player(p)
            team_recv.roster.add_player(p)
            
        # Remove players from receiver and add to proposer
        for p in proposal.receiver_sends:
            team_recv.roster.remove_player(p)
            team_prop.roster.add_player(p)
            
        proposal.status = "Executed"
        
        prop_names = ", ".join(p.name for p in proposal.proposer_sends)
        recv_names = ", ".join(p.name for p in proposal.receiver_sends)
        
        self.transaction_history.append({
            "type": "TRADE",
            "week": self.current_week,
            "team_name": f"{team_prop.name} & {team_recv.name}",
            "player_name": f"{prop_names} <-> {recv_names}",
            "position": "TRADE",
            "details": f"Trade executed: {team_prop.name} sends ({prop_names}), {team_recv.name} sends ({recv_names})."
        })
        return True

    def simulate_week(self):
        """Simulates all head-to-head matchups for the current week."""
        week = self.current_week
        matchups = self.schedule.get(week, [])
        if not matchups:
            raise ValueError(f"No matchups scheduled for Week {week}.")
            
        print(f"Resolving matchups for Week {week}...")
        
        # Query actual stats for all players in this week
        actual_stats = {s.player_id: s.stats for s in db.get_weekly_stats(self.year, week)}
        
        # Query schedule details to resolve DST points allowed
        # Find points scored by opponent for each team
        # We'll scrape schedule scores to calculate points allowed for defenses
        schedule_rows = []
        try:
            schedule_df = nfl.import_schedules([self.year])
            schedule_rows = schedule_df[schedule_df["week"] == week].to_dict("records")
        except Exception as e:
            print(f"Error fetching schedule for DST points allowed: {e}")
            
        pts_allowed_map = {}
        for row in schedule_rows:
            home = row["home_team"]
            away = row["away_team"]
            h_score = row["home_score"]
            a_score = row["away_score"]
            
            # Map opponent points
            if not pd.isna(h_score) and not pd.isna(a_score):
                pts_allowed_map[home] = a_score
                pts_allowed_map[away] = h_score
                
        # Resolve each matchup
        for matchup in matchups:
            team_a = self.teams[matchup.team_a_id]
            team_b = self.teams[matchup.team_b_id]
            
            # Lock rosters for this matchup record
            matchup.team_a_starters = list(team_a.roster.starters)
            matchup.team_b_starters = list(team_b.roster.starters)
            
            # Score Team A
            a_score = 0.0
            for player in matchup.team_a_starters:
                p_stats = actual_stats.get(player.id, {})
                # If player is DST, insert points_allowed from our schedule map
                if player.position == "DST" or (hasattr(player.position, "value") and player.position.value == "DST"):
                    p_stats = dict(p_stats) # copy
                    p_stats["points_allowed"] = pts_allowed_map.get(player.nfl_team, 21.0) # default if bye/not found
                
                a_score += get_player_fantasy_points(player, p_stats, self.settings.scoring)
                
            # Score Team B
            b_score = 0.0
            for player in matchup.team_b_starters:
                p_stats = actual_stats.get(player.id, {})
                if player.position == "DST" or (hasattr(player.position, "value") and player.position.value == "DST"):
                    p_stats = dict(p_stats)
                    p_stats["points_allowed"] = pts_allowed_map.get(player.nfl_team, 21.0)
                    
                b_score += get_player_fantasy_points(player, p_stats, self.settings.scoring)
                
            matchup.team_a_score = round(a_score, 2)
            matchup.team_b_score = round(b_score, 2)
            matchup.completed = True
            
            # Update team records
            team_a.points_for += matchup.team_a_score
            team_a.points_against += matchup.team_b_score
            team_b.points_for += matchup.team_b_score
            team_b.points_against += matchup.team_a_score
            
            if matchup.team_a_score > matchup.team_b_score:
                team_a.wins += 1
                team_b.losses += 1
            elif matchup.team_b_score > matchup.team_a_score:
                team_b.wins += 1
                team_a.losses += 1
            else:
                team_a.ties += 1
                team_b.ties += 1
                
        # Re-sort waiver priority: worst record gets first priority
        # Sort key: wins asc, points_for asc
        self.waiver_priority = sorted(
            list(self.teams.keys()),
            key=lambda tid: (self.teams[tid].wins, self.teams[tid].points_for)
        )
        
        # Advance calendar
        self.current_week += 1

    def get_standings(self) -> List[Team]:
        """Returns a list of teams sorted by standings order (win percentage, then points for)."""
        def standings_key(t: Team):
            total_games = t.wins + t.losses + t.ties
            win_pct = (t.wins + 0.5 * t.ties) / total_games if total_games > 0 else 0.0
            return (win_pct, t.points_for)
            
        return sorted(list(self.teams.values()), key=standings_key, reverse=True)
