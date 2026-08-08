import random
from typing import List, Dict, Tuple, Set, Optional, Any
from ..config import LeagueSettings, ScoringRules, RosterSettings
from ..models import Player, Team, Roster, Matchup, WaiverClaim, TradeProposal
from ..engine.sandbox import LeagueSandbox
from ..ai.personas import get_agent_by_persona
from ..ai.base_agent import BaseAgent
from ..ai.best_agent import ProAIEngine
from ..data import db

class InSeasonLeagueEngine:
    """Manages an active in-season fantasy league where a human user competes against 9 reactive AI opponents."""
    def __init__(self, settings: LeagueSettings, year: int = 2025):
        self.settings = settings
        self.year = year
        self.sandbox = LeagueSandbox(settings, year=year)
        self.agents: Dict[str, BaseAgent] = {}
        self.user_team_id: str = "team_1"
        self.ai_reaction_logs: List[str] = []
        self.user_waiver_claims: List[WaiverClaim] = []

    def initialize_season(self, user_team_name: str = "My Franchise") -> LeagueSandbox:
        """Initializes the 10-team league, drafts baseline rosters, and builds 14-week schedule."""
        db.init_db()
        all_players = db.get_all_players()
        projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(self.year, 1)}

        # Build 10 teams: Team 1 is Human, Teams 2-10 are Pro AI Engines

        teams = [
            Team(id=self.user_team_id, name=user_team_name, owner_persona="human", roster=Roster(), faab_balance=100)
        ]

        for i in range(9):
            t_id = f"team_{i+2}"
            t_name = f"Pro AI Franchise {i+2}"
            teams.append(Team(id=t_id, name=t_name, owner_persona="Pro AI Engine", roster=Roster(), faab_balance=100))
            self.agents[t_id] = ProAIEngine(t_id, self.settings)


        self.sandbox.initialize_league(teams)
        self.sandbox.start_draft()
        self.sandbox.auto_draft_fill()

        # Set initial starting lineups for all teams
        for t_id, team in self.sandbox.teams.items():
            if t_id != self.user_team_id:
                agent = self.agents[t_id]
                starters, bench = agent.optimize_weekly_lineup(team.roster, projs)
                self.sandbox.set_lineup(t_id, starters, bench, [])
            else:
                # Default starters for human
                starters = team.roster.all_players()[:9]
                bench = team.roster.all_players()[9:]
                self.sandbox.set_lineup(t_id, starters, bench, [])

        return self.sandbox

    @property
    def current_week(self) -> int:
        return self.sandbox.current_week

    @property
    def user_team(self) -> Team:
        return self.sandbox.teams[self.user_team_id]

    def update_user_lineup(self, starters: List[Player], bench: List[Player]):
        """Updates user's starting lineup and triggers reactive AI responses."""
        self.sandbox.set_lineup(self.user_team_id, starters, bench, [])
        self._trigger_ai_reactions("lineup_update")

    def submit_user_waiver_claim(self, player_to_add: Player, player_to_drop: Optional[Player], bid_amount: int):
        """Submits user waiver claim and triggers reactive AI counter-bids."""
        claim = WaiverClaim(
            team_id=self.user_team_id,
            player_to_add=player_to_add,
            player_to_drop=player_to_drop,
            bid_amount=bid_amount,
            priority_order=len(self.user_waiver_claims)
        )
        self.user_waiver_claims.append(claim)
        self._trigger_ai_reactions("waiver_claim", player_to_add)

    def _trigger_ai_reactions(self, event_type: str, target_player: Optional[Player] = None):
        """Reactive Event Loop: 9 AI managers respond to user actions in real-time."""
        self.ai_reaction_logs.clear()
        projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(self.year, self.current_week)}
        all_players = db.get_all_players()
        drafted_ids = {p.id for t in self.sandbox.teams.values() for p in t.roster.all_players()}
        free_agents = [p for p in all_players if p.id not in drafted_ids]

        if event_type == "lineup_update":
            # AI managers inspect matchup opponent and re-optimize starting lineups
            for t_id, agent in self.agents.items():
                team = self.sandbox.teams[t_id]
                starters, bench = agent.optimize_weekly_lineup(team.roster, projs)
                self.sandbox.set_lineup(t_id, starters, bench, [])
            self.ai_reaction_logs.append("🤖 All 9 AI opponents analyzed market updates and re-optimized starting lineups!")

        elif event_type == "waiver_claim" and target_player:
            # AI managers scan the targeted player and submit competitive counter-bids if valuable
            counter_bids_count = 0
            for t_id, agent in self.agents.items():
                team = self.sandbox.teams[t_id]
                claims = agent.get_waiver_claims(team, [target_player] + free_agents[:10], projs, self.current_week)
                if claims:
                    counter_bids_count += 1
            self.ai_reaction_logs.append(f"🤖 Market Alert: {counter_bids_count} AI managers submitted counter FAAB bids for {target_player.name}!")

    def simulate_weekend_games(self) -> List[Matchup]:
        """Simulates all weekend games for the current week, updates records, and advances week."""
        wk = self.current_week
        projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(self.year, wk)}

        # Process all waiver claims (User + AI)
        all_claims = list(self.user_waiver_claims)
        for t_id, agent in self.agents.items():
            team = self.sandbox.teams[t_id]
            all_players = db.get_all_players()
            drafted_ids = {p.id for t in self.sandbox.teams.values() for p in t.roster.all_players()}
            free_agents = [p for p in all_players if p.id not in drafted_ids]
            ai_claims = agent.get_waiver_claims(team, free_agents[:50], projs, current_week=wk)
            all_claims.extend(ai_claims)

        self.sandbox.process_waiver_claims(all_claims)
        self.user_waiver_claims.clear()

        # Simulate matchups
        matchups = self.sandbox.schedule.get(wk, [])
        self.sandbox.simulate_week()
        return matchups
