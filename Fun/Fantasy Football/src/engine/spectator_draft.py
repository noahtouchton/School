import random
from typing import List, Dict, Tuple, Set, Optional, Any
from ..config import LeagueSettings, ScoringRules, RosterSettings
from ..models import Player, Team, Roster, DraftState
from ..engine.sandbox import LeagueSandbox
from ..ai.personas import get_agent_by_persona
from ..ai.base_agent import BaseAgent
from ..data import db

DEFAULT_SPECTATOR_PERSONAS = [
    "zero_rb", "hero_rb", "robust_rb", "late_round_qb", "high_risk",
    "conservative", "trade_demon", "free_agent_demon", "matchup_all_star", "balanced"
]

from ..ai.best_agent import ProAIEngine, get_pro_ai_agent

class SpectatorDraftEngine:
    """Manages a 10-AI live spectator draft room where 10 Pro AI engines draft against each other pick-by-pick."""
    def __init__(self, settings: LeagueSettings, year: int):
        self.settings = settings
        self.year = year
        self.sandbox = LeagueSandbox(settings, year=year)
        self.agents: Dict[str, BaseAgent] = {}
        self.projections: Dict[str, float] = {}
        self.all_players: List[Player] = []
        self.pick_logs: List[Dict[str, Any]] = []

    def initialize_draft(self) -> DraftState:
        """Sets up 10 Pro AI teams, applies session draft variance, randomizes draft order, and starts draft."""
        db.init_db()
        self.all_players = db.get_all_players()
        
        # Load base projections and apply stochastic session variance so every draft evaluates players uniquely
        base_projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(self.year, 1)}
        self.projections = {
            p_id: round(pts * random.uniform(0.88, 1.12), 2)
            for p_id, pts in base_projs.items()
        }

        teams = []
        for i in range(10):
            t_id = f"team_{i+1}"
            # Create Pro AI engine with randomized strategy flavor (varied stack boost, decay, and rookie weights)
            from ..ai.base_agent import AgentParameters
            custom_params = AgentParameters(
                vorp_decay_qb=round(random.uniform(0.40, 0.70), 2),
                vorp_decay_rb=round(random.uniform(0.35, 0.60), 2),
                vorp_decay_wr=round(random.uniform(0.35, 0.60), 2),
                vorp_decay_te=round(random.uniform(0.25, 0.50), 2),
                rookie_boost=round(random.uniform(0.95, 1.30), 2),
                age_penalty_threshold=random.choice([27, 28, 29, 30, 31]),
                qb_wr_stack_boost=round(random.uniform(1.05, 1.35), 2),
                matchup_adjustment=round(random.uniform(0.50, 1.00), 2)
            )
            agent = BaseAgent(t_id, self.settings, custom_params)
            
            teams.append(Team(
                id=t_id,
                name=f"Pro AI Franchise {i+1}",
                owner_persona="Pro AI Engine",
                roster=Roster(),
                faab_balance=100
            ))
            self.agents[t_id] = agent

        # Randomize team draft slot order every time
        random.shuffle(teams)

        self.sandbox.initialize_league(teams)
        return self.sandbox.start_draft()


    @property
    def is_complete(self) -> bool:
        if not self.sandbox.draft_state:
            return False
        total_picks = self.settings.roster.total_roster_spots() * len(self.sandbox.teams)
        return len(self.sandbox.draft_state.picks) >= total_picks

    def step_next_pick(self) -> Optional[Dict[str, Any]]:
        """Executes the next single pick in the draft order, computing dynamic VORP & scarcity math."""
        if self.is_complete or not self.sandbox.draft_state:
            return None

        ds = self.sandbox.draft_state
        current_team_id = ds.get_current_team_id()
        agent = self.agents[current_team_id]
        team = self.sandbox.teams[current_team_id]

        undrafted = [p for p in self.all_players if p.id not in ds.drafted_player_ids]
        if not undrafted:
            return None

        # Execute dynamic AI math decision
        selected_player = agent.draft_pick(ds, undrafted, self.projections)
        self.sandbox.execute_draft_pick(selected_player)

        pick_num = len(ds.picks)
        round_num = ds.current_round
        pos_str = selected_player.position.value if hasattr(selected_player.position, "value") else str(selected_player.position)
        proj_pts = self.projections.get(selected_player.id, 0.0)

        vorp_dict = agent.get_vorp_scores(undrafted, self.projections)
        vorp_val = round(vorp_dict.get(selected_player.id, 0.0), 1)

        # Dynamic scarcity math
        pos_remaining = len([p for p in undrafted if (p.position.value if hasattr(p.position, "value") else str(p.position)) == pos_str])
        
        reason = f"Evaluated Math: Proj: {proj_pts} pts/wk | Baseline VORP: +{vorp_val} | Pos Scarcity ({pos_str}): {pos_remaining} left"

        pick_info = {
            "pick_number": pick_num,
            "round": round_num,
            "team_id": current_team_id,
            "team_name": team.name,
            "persona": team.owner_persona,
            "player": selected_player,
            "position": pos_str,
            "projected": proj_pts,
            "vorp": vorp_val,
            "reasoning": reason
        }
        self.pick_logs.append(pick_info)
        return pick_info


    def auto_complete_draft(self):
        """Fast-forwards and completes all remaining draft picks."""
        while not self.is_complete:
            self.step_next_pick()
