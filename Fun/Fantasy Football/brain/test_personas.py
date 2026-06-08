import sys
import os

# Add src to the path
sys.path.append("/Users/noahtouchton/School_Git/School/Fun/Fantasy Football")

from src.config import LeagueSettings, ScoringRules, RosterSettings
from src.models import Team, Roster
from src.engine.sandbox import LeagueSandbox
from src.ai.personas import get_agent_by_persona
from src.data import db

def run_test():
    print("Loading database...")
    db.init_db()
    
    # 1. Setup League Settings
    settings = LeagueSettings(
        name="AI Clash Sandbox League",
        teams_count=8, # 8 teams matching our 8 distinct personas
        scoring=ScoringRules.half_ppr(),
        roster=RosterSettings(qb=1, rb=2, wr=2, te=1, flex=2, bench=6) # 12 roster spots
    )
    
    sandbox = LeagueSandbox(settings, year=2024)
    
    # 2. Create 8 Teams with their specific personas
    personas = [
        "free_agent_demon", "trade_demon", "matchup_all_star", "conservative",
        "zero_rb", "hero_rb", "high_risk", "balanced"
    ]
    teams = []
    agents = {}
    for i, p_name in enumerate(personas):
        t_id = f"team_{i+1}"
        team = Team(
            id=t_id,
            name=f"Team {p_name.replace('_', ' ').title()}",
            owner_persona=p_name,
            roster=Roster(),
            faab_balance=100
        )
        teams.append(team)
        agents[t_id] = get_agent_by_persona(p_name, t_id, settings)
        
    print("Initializing league...")
    sandbox.initialize_league(teams)
    
    # 3. Start Draft
    print("\nStarting Draft...")
    draft_state = sandbox.start_draft()
    all_players = db.get_all_players()
    projs = {p.player_id: p.projected_points for p in db.get_weekly_projections(2024, 1)}
    
    # Run Draft round-by-round
    total_rounds = settings.roster.total_roster_spots()
    total_picks = total_rounds * len(teams)
    
    print(f"Drafting {total_picks} picks over {total_rounds} rounds...")
    for pick_idx in range(total_picks):
        current_team_id = draft_state.get_current_team_id()
        agent = agents[current_team_id]
        
        # Get undrafted players
        undrafted = [p for p in all_players if p.id not in draft_state.drafted_player_ids]
        
        # Agent makes their choice
        selected_player = agent.draft_pick(draft_state, undrafted, projs)
        sandbox.execute_draft_pick(selected_player)
        
    print("Draft completed!")
    
    # Let's inspect Zero-RB and Hero-RB rosters to see if strategies worked!
    print("\nChecking Roster Strategy Composition:")
    for t_id, team in sandbox.teams.items():
        if "Zero Rb" in team.name or "Hero Rb" in team.name:
            print(f"\nRoster for {team.name}:")
            # We want to show which picks were drafted. Let's see the draft picks for this team
            team_picks = [p for p in draft_state.picks if p["team_id"] == t_id]
            for p in team_picks:
                player = p["player"]
                pos = player.position.value if hasattr(player.position, "value") else str(player.position)
                print(f"  Round {p['round']}: {player.name} ({pos}) - {player.nfl_team}")

    # 4. Set Lineups for Week 1
    print("\nOptimizing week 1 lineups...")
    for t_id, team in sandbox.teams.items():
        agent = agents[t_id]
        starters, bench = agent.optimize_weekly_lineup(team.roster, projs)
        sandbox.set_lineup(t_id, starters, bench, [])
        
    # 5. Waiver claims run (Week 1 waivers are run after draft)
    print("\nEvaluating waiver wire claims for Week 1...")
    all_claims = []
    # Make a mock free agent pool (let's say 20 players who are undrafted)
    undrafted_pool = [p for p in all_players if p.id not in draft_state.drafted_player_ids]
    for t_id, team in sandbox.teams.items():
        agent = agents[t_id]
        team_claims = agent.get_waiver_claims(team, undrafted_pool[:50], projs, current_week=1)
        all_claims.extend(team_claims)
        
    print(f"Total waiver claims submitted: {len(all_claims)}")
    if all_claims:
        sandbox.process_waiver_claims(all_claims)
        
    # 6. Trade Period (Trade Demon proposes)
    print("\nEvaluating trade proposals...")
    trade_proposals = []
    for t_id, team in sandbox.teams.items():
        agent = agents[t_id]
        # Only trade demon will generate proposals in our simple loop
        if hasattr(agent, "generate_trade_proposals"):
            proposals = agent.generate_trade_proposals(team, list(sandbox.teams.values()), projs)
            trade_proposals.extend(proposals)
            
    print(f"Trade Demon generated {len(trade_proposals)} trade proposals.")
    for proposal in trade_proposals:
        recv_agent = agents[proposal.receiver_team_id]
        recv_team = sandbox.teams[proposal.receiver_team_id]
        # Evaluate
        accepted = recv_agent.evaluate_trade_proposal(recv_team, proposal, projs)
        proposer_team = sandbox.teams[proposal.proposer_team_id]
        give_names = ", ".join(p.name for p in proposal.proposer_sends)
        recv_names = ", ".join(p.name for p in proposal.receiver_sends)
        print(f"- Proposal: {proposer_team.name} sends {give_names} <-> {recv_team.name} sends {recv_names}")
        print(f"  Response: {'ACCEPTED' if accepted else 'REJECTED'}")
        if accepted:
            sandbox.execute_trade(proposal)
            
    # 7. Simulate Week 1
    print("\nSimulating Week 1...")
    sandbox.simulate_week()
    
    # 8. Display standins
    print("\nStandings after Week 1:")
    standings = sandbox.get_standings()
    for idx, team in enumerate(standings):
        print(f"{idx+1}. {team.name} ({team.record_str}) PF: {team.points_for:.2f} PA: {team.points_against:.2f} FAAB: ${team.faab_balance}")

if __name__ == "__main__":
    run_test()
