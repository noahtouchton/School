import random
import copy
import os
import sys
from typing import List, Dict, Tuple, Any, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from .base_agent import BaseAgent, AgentParameters
from ..config import LeagueSettings, ScoringRules, RosterSettings
from ..models import Team, Roster
from ..engine.sandbox import LeagueSandbox
from ..data import db

def run_single_season_eval_against_personas(year: int, candidate_params: AgentParameters, opponent_styles: List[str]) -> float:
    """Runs a single season simulation where the candidate agent competes against 9 other AIs.
    Returns the fitness score of the candidate team: (Wins * 100) + Points For.
    """
    # SQLite requires connections to be per-process/thread
    db.init_db()
    
    # Imports inside function to avoid circular dependencies
    from src.ai.personas import get_agent_by_persona
    
    settings = LeagueSettings(
        name="GA Persona Eval League",
        teams_count=10,
        scoring=ScoringRules.half_ppr(),
        roster=RosterSettings(qb=1, rb=2, wr=2, te=1, flex=2, bench=6)
    )
    
    sandbox = LeagueSandbox(settings, year=year)
    
    # Create Teams and Agents
    # Team 1 is the candidate
    teams = [
        Team(
            id="team_candidate",
            name="Candidate Agent",
            owner_persona="parameterized",
            roster=Roster(),
            faab_balance=100
        )
    ]
    agents = {
        "team_candidate": BaseAgent("team_candidate", settings, candidate_params)
    }
    
    # Other teams are opponent personas
    for i, style in enumerate(opponent_styles):
        t_id = f"team_opponent_{i+1}"
        teams.append(Team(
            id=t_id,
            name=f"Opponent {style.replace('_', ' ').title()}",
            owner_persona=style,
            roster=Roster(),
            faab_balance=100
        ))
        agents[t_id] = get_agent_by_persona(style, t_id, settings)
        
    sandbox.initialize_league(teams)
    
    # 1. Draft
    draft_state = sandbox.start_draft()
    all_players = db.get_all_players()
    # Query projections for the year, week 1 (for draft ADP purposes)
    projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(year, 1)}
    
    total_picks = settings.roster.total_roster_spots() * len(teams)
    for _ in range(total_picks):
        t_id = draft_state.get_current_team_id()
        agent = agents[t_id]
        undrafted = [p for p in all_players if p.id not in draft_state.drafted_player_ids]
        
        selected = agent.draft_pick(draft_state, undrafted, projs)
        sandbox.execute_draft_pick(selected)
        
    # 2. Week-by-Week Season Loop
    num_weeks = 14
    for week in range(1, num_weeks + 1):
        weekly_projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(year, week)}
        
        # Lineups
        for t_id, team in sandbox.teams.items():
            agent = agents[t_id]
            starters, bench = agent.optimize_weekly_lineup(team.roster, weekly_projs)
            sandbox.set_lineup(t_id, starters, bench, [])
            
        # Waivers
        free_agents = [p for p in all_players if p.id not in sandbox.draft_state.drafted_player_ids]
        random.shuffle(free_agents)
        
        all_claims = []
        for t_id, team in sandbox.teams.items():
            agent = agents[t_id]
            claims = agent.get_waiver_claims(team, free_agents[:50], weekly_projs, current_week=week)
            all_claims.extend(claims)
        sandbox.process_waiver_claims(all_claims)
        
        # Trades - ALL agents actively scan and trade
        trade_proposals = []
        for t_id, team in sandbox.teams.items():
            agent = agents[t_id]
            proposals = agent.generate_trade_proposals(team, list(sandbox.teams.values()), weekly_projs)
            trade_proposals.extend(proposals)
        
        for proposal in trade_proposals:
            recv_agent = agents[proposal.receiver_team_id]
            recv_team = sandbox.teams[proposal.receiver_team_id]
            if recv_agent.evaluate_trade_proposal(recv_team, proposal, weekly_projs):
                sandbox.execute_trade(proposal)
                
        # Matchups
        sandbox.simulate_week()
        
    # 3. Calculate Scores: (Wins * 100) + Points For
    candidate_team = sandbox.teams["team_candidate"]
    score = (candidate_team.wins * 100.0) + candidate_team.points_for
    return score


def run_single_season_eval(year: int, params_list: List[AgentParameters]) -> List[float]:
    """Helper function to run a single sandbox season simulation.
    Must be a top-level module function to be pickled and run in parallel processes.
    """
    # SQLite requires connections to be per-process/thread
    db.init_db()
    
    settings = LeagueSettings(
        name="GA Eval League",
        teams_count=len(params_list),
        scoring=ScoringRules.half_ppr(),
        roster=RosterSettings(qb=1, rb=2, wr=2, te=1, flex=2, bench=6)
    )
    
    sandbox = LeagueSandbox(settings, year=year)
    
    # Create Teams and Agents
    teams = []
    agents = {}
    for i, params in enumerate(params_list):
        t_id = f"team_{i+1}"
        teams.append(Team(
            id=t_id,
            name=f"Agent {i+1}",
            owner_persona="parameterized",
            roster=Roster(),
            faab_balance=100
        ))
        agents[t_id] = BaseAgent(t_id, settings, params)
        
    sandbox.initialize_league(teams)
    
    # 1. Draft
    draft_state = sandbox.start_draft()
    all_players = db.get_all_players()
    # Query projections for the year, week 1 (for draft ADP purposes)
    projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(year, 1)}
    
    total_picks = settings.roster.total_roster_spots() * len(teams)
    for _ in range(total_picks):
        t_id = draft_state.get_current_team_id()
        agent = agents[t_id]
        undrafted = [p for p in all_players if p.id not in draft_state.drafted_player_ids]
        
        selected = agent.draft_pick(draft_state, undrafted, projs)
        sandbox.execute_draft_pick(selected)
        
    # 2. Week-by-Week Season Loop
    num_weeks = 14
    for week in range(1, num_weeks + 1):
        weekly_projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(year, week)}
        
        # Lineups
        for t_id, team in sandbox.teams.items():
            agent = agents[t_id]
            starters, bench = agent.optimize_weekly_lineup(team.roster, weekly_projs)
            sandbox.set_lineup(t_id, starters, bench, [])
            
        # Waivers
        free_agents = [p for p in all_players if p.id not in sandbox.draft_state.drafted_player_ids]
        random.shuffle(free_agents)
        
        all_claims = []
        for t_id, team in sandbox.teams.items():
            agent = agents[t_id]
            claims = agent.get_waiver_claims(team, free_agents[:50], weekly_projs, current_week=week)
            all_claims.extend(claims)
        sandbox.process_waiver_claims(all_claims)
        
        # Trades - ALL agents actively scan and trade
        trade_proposals = []
        for t_id, team in sandbox.teams.items():
            agent = agents[t_id]
            proposals = agent.generate_trade_proposals(team, list(sandbox.teams.values()), weekly_projs)
            trade_proposals.extend(proposals)
        
        for proposal in trade_proposals:
            recv_agent = agents[proposal.receiver_team_id]
            recv_team = sandbox.teams[proposal.receiver_team_id]
            if recv_agent.evaluate_trade_proposal(recv_team, proposal, weekly_projs):
                sandbox.execute_trade(proposal)
                
        # Matchups
        sandbox.simulate_week()
        
    # 3. Calculate Scores: (Wins * 100) + Points For
    scores = []
    for team in teams:
        team_final = sandbox.teams[team.id]
        score = (team_final.wins * 100.0) + team_final.points_for
        scores.append(score)
        
    return scores


class EvolutionaryTrainer:
    """Trains fantasy football AI agents using parallel genetic algorithms.
    Evaluates agents over random seasons and shuffles, preserving playstyle constraints.
    """
    def __init__(self, population_size: int = 20, cached_years: Optional[List[int]] = None):
        self.population_size = population_size
        self.cached_years = cached_years or [2025, 2024, 2023]
        self.population: List[AgentParameters] = []
        self.ALL_PLAYSTYLES = [
            "balanced", "free_agent_demon", "trade_demon", "matchup_all_star",
            "conservative", "zero_rb", "hero_rb", "high_risk", "late_round_qb", "robust_rb"
        ]

    def get_opponent_styles(self, target_playstyle: str) -> List[str]:
        """Returns the 9 opponent styles to play against the target style in a 10-team league."""
        target_clean = target_playstyle.lower().replace("_", "")
        matched_target = None
        for style in self.ALL_PLAYSTYLES:
            if style.lower().replace("_", "") == target_clean:
                matched_target = style
                break
                
        if matched_target and matched_target in self.ALL_PLAYSTYLES:
            opponents = [s for s in self.ALL_PLAYSTYLES if s != matched_target]
            return opponents
        else:
            # For hybrid, return 9 random selections from the 10 playstyles
            opponents = list(self.ALL_PLAYSTYLES)
            random.shuffle(opponents)
            return opponents[:9]
        
    def initialize_population(self, playstyle: str = "hybrid"):
        """Seeds the population. If training a playstyle, forces constraints on the seeds."""
        self.population = []
        
        # Create standard starting template
        template = self.get_playstyle_template(playstyle)
        
        for _ in range(self.population_size):
            # Start with template and add minor random perturbations
            mutated = self.mutate(template, rate=0.4, playstyle=playstyle)
            self.population.append(mutated)

    def get_playstyle_template(self, playstyle: str) -> AgentParameters:
        """Returns the base parameters for the 10 playstyles or hybrid."""
        style = playstyle.lower().replace("_", "")
        
        if style == "balanced":
            return AgentParameters()
        elif style == "freeagentdemon":
            return AgentParameters(waiver_min_improvement=0.5, waiver_max_faab_pct=0.20)
        elif style == "tradedemon":
            return AgentParameters(trade_min_gain=0.1)
        elif style == "matchupallstar":
            return AgentParameters(matchup_adjustment=1.0)
        elif style == "conservative":
            return AgentParameters(waiver_min_improvement=4.0, waiver_max_faab_pct=0.01, trade_min_gain=2.0, rookie_boost=0.85, age_penalty_factor=0.02)
        elif style == "zerorb":
            return AgentParameters(early_rb_limit=0)
        elif style == "herorb":
            return AgentParameters(early_rb_limit=1)
        elif style == "highrisk":
            return AgentParameters(rookie_boost=1.15, young_player_trade_boost=0.15, waiver_min_improvement=1.0)
        elif style == "lateroundqb":
            return AgentParameters(early_qb_limit=0)
        elif style == "robustrb":
            return AgentParameters(early_rb_minimum=3)
        else: # hybrid
            return AgentParameters()

    def crossover(self, parent_a: AgentParameters, parent_b: AgentParameters, playstyle: str = "hybrid") -> AgentParameters:
        """Combines parameters from two parents, preserving playstyle constraints."""
        child = AgentParameters()
        fields = [
            "vorp_decay_qb", "vorp_decay_rb", "vorp_decay_wr", "vorp_decay_te",
            "early_rb_limit", "early_qb_limit", "early_rb_minimum",
            "rookie_boost", "age_penalty_threshold", "age_penalty_factor",
            "waiver_min_improvement", "waiver_max_faab_pct", "trade_min_gain",
            "young_player_trade_boost", "trade_future_discount",
            "matchup_adjustment", "qb_wr_stack_boost", "faab_urgency_factor"
        ]
        
        # Determine frozen constraints for this playstyle
        frozen = self.get_frozen_fields(playstyle)
        
        for field in fields:
            if field in frozen:
                # Keep the template value
                template = self.get_playstyle_template(playstyle)
                setattr(child, field, getattr(template, field))
            else:
                # Crossover
                val = getattr(parent_a, field) if random.random() < 0.5 else getattr(parent_b, field)
                setattr(child, field, val)
                
        return child

    def mutate(self, params: AgentParameters, rate: float = 0.20, playstyle: str = "hybrid") -> AgentParameters:
        """Mutates numerical fields, leaving playstyle constraints frozen."""
        mutated = copy.deepcopy(params)
        frozen = self.get_frozen_fields(playstyle)
        
        def perturb(field: str, val: float, low: float, high: float) -> float:
            if field in frozen:
                return val
            if random.random() < rate:
                val += random.normalvariate(0, 0.1 * (high - low))
                return max(low, min(high, val))
            return val

        mutated.vorp_decay_qb = perturb("vorp_decay_qb", mutated.vorp_decay_qb, 0.1, 1.0)
        mutated.vorp_decay_rb = perturb("vorp_decay_rb", mutated.vorp_decay_rb, 0.1, 1.0)
        mutated.vorp_decay_wr = perturb("vorp_decay_wr", mutated.vorp_decay_wr, 0.1, 1.0)
        mutated.vorp_decay_te = perturb("vorp_decay_te", mutated.vorp_decay_te, 0.1, 1.0)
        
        if "early_rb_limit" not in frozen and random.random() < rate:
            mutated.early_rb_limit = random.choice([0, 1, 99])
        if "early_qb_limit" not in frozen and random.random() < rate:
            mutated.early_qb_limit = random.choice([0, 99])
        if "early_rb_minimum" not in frozen and random.random() < rate:
            mutated.early_rb_minimum = random.choice([0, 3])
            
        mutated.rookie_boost = perturb("rookie_boost", mutated.rookie_boost, 0.5, 1.5)
        
        if "age_penalty_threshold" not in frozen and random.random() < rate:
            mutated.age_penalty_threshold = random.randint(26, 32)
            
        mutated.age_penalty_factor = perturb("age_penalty_factor", mutated.age_penalty_factor, 0.0, 0.1)
        mutated.waiver_min_improvement = perturb("waiver_min_improvement", mutated.waiver_min_improvement, 0.0, 5.0)
        mutated.waiver_max_faab_pct = perturb("waiver_max_faab_pct", mutated.waiver_max_faab_pct, 0.0, 0.5)
        mutated.trade_min_gain = perturb("trade_min_gain", mutated.trade_min_gain, 0.0, 5.0)
        mutated.young_player_trade_boost = perturb("young_player_trade_boost", mutated.young_player_trade_boost, 0.0, 0.5)
        mutated.trade_future_discount = perturb("trade_future_discount", mutated.trade_future_discount, 0.5, 1.0)
        mutated.matchup_adjustment = perturb("matchup_adjustment", mutated.matchup_adjustment, 0.0, 1.0)
        mutated.qb_wr_stack_boost = perturb("qb_wr_stack_boost", mutated.qb_wr_stack_boost, 0.8, 1.5)
        mutated.faab_urgency_factor = perturb("faab_urgency_factor", mutated.faab_urgency_factor, 0.5, 2.0)
        
        return mutated

    def get_frozen_fields(self, playstyle: str) -> Set[str]:
        """Returns the fields that must remain frozen for a given playstyle."""
        style = playstyle.lower().replace("_", "")
        
        if style == "zerorb":
            return {"early_rb_limit"}
        elif style == "herorb":
            return {"early_rb_limit"}
        elif style == "lateroundqb":
            return {"early_qb_limit"}
        elif style == "robustrb":
            return {"early_rb_minimum"}
        elif style == "freeagentdemon":
            return {"waiver_min_improvement", "waiver_max_faab_pct"}
        elif style == "tradedemon":
            return {"trade_min_gain"}
        elif style == "matchupallstar":
            return {"matchup_adjustment"}
        elif style == "conservative":
            return {"waiver_min_improvement", "waiver_max_faab_pct", "trade_min_gain", "rookie_boost", "age_penalty_factor"}
        elif style == "highrisk":
            return {"rookie_boost", "young_player_trade_boost", "waiver_min_improvement"}
        else:
            return set()

    def evaluate_fitness_parallel(self, params_list: List[AgentParameters], 
                                  seasons_per_eval: int = 5) -> List[float]:
        """Runs batch season simulations in parallel and aggregates fitness scores.
        Distributes work over processes to scale to thousands of seasons.
        """
        # We need to run len(params_list) agents in groups of 10.
        # If params_list size is 20, we can split into 2 groups of 10.
        group_size = 10
        num_groups = len(params_list) // group_size
        
        if num_groups == 0:
            raise ValueError("Population size must be at least 10 to form a standard league.")
            
        # We'll run `seasons_per_eval` simulated seasons for each group.
        # Each simulation uses a randomly selected year and shuffles draft order.
        futures = []
        
        # Prepare parallel executor
        # Limit max workers to prevent crashing low-resource environments
        max_workers = min(os.cpu_count() or 4, 8)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # For each group and each season, launch a simulation task
            for group_idx in range(num_groups):
                start = group_idx * group_size
                group_params = params_list[start:start+group_size]
                
                for _ in range(seasons_per_eval):
                    # Random year selection
                    year = random.choice(self.cached_years)
                    # Shuffle parameters list to randomize draft order!
                    shuffled_params = list(group_params)
                    random.shuffle(shuffled_params)
                    
                    # Store original index map so we can map results back to individual agents
                    index_map = {param: group_params.index(param) for param in shuffled_params}
                    
                    futures.append(
                        (executor.submit(run_single_season_eval, year, shuffled_params), start, index_map)
                    )
            
            # Aggregate fitness scores
            # scores_accumulator[pop_index] = list of scores across simulated seasons
            scores_accumulator: Dict[int, List[float]] = {i: [] for i in range(len(params_list))}
            
            for future, group_start_idx, idx_map in futures:
                try:
                    # scores returned from single_season match the order of shuffled_params passed to it
                    shuffled_scores = future.result()
                    
                    # Map scores back to original indices in params_list
                    for shuffled_idx, score in enumerate(shuffled_scores):
                        param = idx_map.get(shuffled_scores) # wait, we passed shuffled_params to submit.
                        # Let's get the original parameter object
                        # The future was run with shuffled_params. So shuffled_scores corresponds to shuffled_params!
                        # Let's find what parameter it was
                        # Wait, we can index shuffled_params directly:
                        # shuffled_params is not directly accessible here unless we stored it in the future tuple!
                        # Yes, we can adjust the submission to store the shuffled_params.
                        pass
                except Exception as e:
                    print(f"Simulation process encountered an error: {e}")
                    
        # Let's write the mapping logic cleanly.
        # A simpler, bulletproof parallel structure:
        # Each process runs a list of seasons for a single group, and returns the aggregated scores!
        # This keeps the parameters and mapping self-contained in the process loop.
        
        return self._evaluate_fitness_parallel_clean(params_list, seasons_per_eval)

    def _evaluate_fitness_parallel_clean(self, params_list: List[AgentParameters], 
                                         seasons_per_eval: int = 5,
                                         target_playstyle: str = "hybrid") -> List[float]:
        """A clean, parallel evaluator that evaluates each candidate against the other 9 personas.
        Returns a list of fitness scores matching the input params_list.
        """
        # Determine opponent styles for this training run
        opponent_styles = self.get_opponent_styles(target_playstyle)
        
        futures = []
        max_workers = min(os.cpu_count() or 4, 8)
        
        # We submit seasons_per_eval simulation tasks for EACH candidate
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for idx, candidate_params in enumerate(params_list):
                for _ in range(seasons_per_eval):
                    year = random.choice(self.cached_years)
                    # Submit a single season simulation
                    futures.append((executor.submit(run_single_season_eval_against_personas, year, candidate_params, opponent_styles), idx))
            
            # Accumulate scores per candidate
            scores_accum = {i: [] for i in range(len(params_list))}
            
            for future, idx in futures:
                try:
                    score = future.result()
                    scores_accum[idx].append(score)
                except Exception as e:
                    print(f"Error in parallel simulation for candidate {idx}: {e}")
                    
        # Calculate average fitness score for each candidate agent
        final_scores = []
        for i in range(len(params_list)):
            agent_scores = scores_accum.get(i, [])
            avg_score = sum(agent_scores) / len(agent_scores) if agent_scores else 0.0
            final_scores.append(avg_score)
            
        return final_scores

    def train_playstyle(self, playstyle: str = "hybrid", generations: int = 5, 
                        seasons_per_eval: int = 5, progress_callback = None) -> AgentParameters:
        """Runs the evolutionary training loop for a specific playstyle preset or hybrid.
        Frozens playstyle-specific constraints and optimizes all other parameters.
        """
        self.initialize_population(playstyle)
        
        print(f"🧬 Starting Training for playstyle: '{playstyle}'...")
        print(f"Generations: {generations}, Seasons per eval: {seasons_per_eval}")
        
        for g in range(1, generations + 1):
            if progress_callback:
                progress_callback(g, generations, f"Simulating Generation {g}/{generations}...")
                
            # Randomize order
            random.shuffle(self.population)
            
            # Evaluate fitness in parallel
            fitness_scores = self._evaluate_fitness_parallel_clean(self.population, seasons_per_eval, playstyle)
            
            # Map
            fitness_map = {idx: score for idx, score in enumerate(fitness_scores)}
            sorted_indices = sorted(list(fitness_map.keys()), key=lambda idx: fitness_map[idx], reverse=True)
            sorted_pop = [self.population[idx] for idx in sorted_indices]
            
            top_score = fitness_map[sorted_indices[0]]
            avg_score = sum(fitness_map.values()) / len(fitness_map)
            
            print(f"Gen {g} - Top Fitness: {top_score:.2f}, Avg: {avg_score:.2f}")
            
            # Keep top 50%
            survivors = sorted_pop[:self.population_size // 2]
            
            # Generate offspring
            offspring = []
            while len(survivors) + len(offspring) < self.population_size:
                parent_a = random.choice(survivors)
                parent_b = random.choice(survivors)
                
                child = self.crossover(parent_a, parent_b, playstyle=playstyle)
                child = self.mutate(child, playstyle=playstyle)
                offspring.append(child)
                
            self.population = survivors + offspring
            
        # Return best parameter set from final generation
        best_agent = self.population[0]
        
        # Save to database permanently
        # Map playstyle name to a pretty db model name
        model_name = f"Optimized {playstyle.replace('_', ' ').title()}"
        db.save_trained_model(model_name, best_agent.__dict__)
        print(f"💾 Permanently saved evolved model '{model_name}' to SQLite database!")
        
        return best_agent
