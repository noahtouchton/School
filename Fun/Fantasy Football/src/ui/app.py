import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import random
import json

# Ensure workspace is on python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import LeagueSettings, ScoringRules, RosterSettings
from src.models import Team, Roster, Player, Position
from src.engine.sandbox import LeagueSandbox
from src.ai.personas import get_agent_by_persona
from src.ai.base_agent import BaseAgent
from src.ai.evolutionary import EvolutionaryTrainer
from src.data import db

# Set page config for SEO and layout
st.set_page_config(
    page_title="Antigravity Fantasy Football AI Simulator",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Main container styling */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek gradient background header */
    .header-container {
        background: linear-gradient(135deg, #3a1c71, #d76d77, #ffaf7b);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }
    
    .header-container h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Glassmorphism Cards */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
    }
    
    /* Position Badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        color: white;
        margin-right: 0.5rem;
    }
    
    .badge-qb { background-color: #2e62ff; }
    .badge-rb { background-color: #1aa160; }
    .badge-wr { background-color: #8c2eff; }
    .badge-te { background-color: #ff7700; }
    .badge-k { background-color: #ff3377; }
    .badge-dst { background-color: #6c757d; }
    
</style>
""", unsafe_allow_html=True)

# Helper to fetch players and projections once
@st.cache_data
def get_cached_players():
    db.init_db()
    return db.get_all_players()

@st.cache_data
def get_cached_projs(year, week):
    return {p.player_id: p.projected_points for p in db.get_weekly_projections(year, week)}

# Main app title
st.markdown("""
<div class="header-container" id="main-header">
    <h1>🏈 Fantasy Football AI Simulator</h1>
    <p>Sandbox Testing, Custom Strategy Draft Rooms & Evolutionary AI Training Grounds</p>
</div>
""", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.markdown("## ⚙️ League Presets")
league_size = st.sidebar.selectbox("Teams Count", [8, 10, 12], index=1)
scoring_type = st.sidebar.selectbox("Scoring Rules", ["PPR (1.0)", "Half-PPR (0.5)", "Standard (0.0)", "TE Premium (1.5)"])
sim_year = st.sidebar.selectbox("NFL Reference Season", [2025, 2024, 2023, 2022], index=0)

# Check if selected year is cached. If not, trigger scrape!
if not db.is_year_cached(sim_year, "stats") or not db.is_year_cached(sim_year, "rosters"):
    from src.data import scraper
    with st.spinner(f"🏈 Initializing database for the {sim_year} season. Scraping rosters, play statistics, and generating projections (takes ~15s)..."):
        scraper.scrape_and_cache_season(sim_year)
    st.success(f"Successfully scraped and cached the {sim_year} season!")
    st.rerun()

# Build configurations based on inputs
scoring_map = {
    "PPR (1.0)": ScoringRules.ppr(),
    "Half-PPR (0.5)": ScoringRules.half_ppr(),
    "Standard (0.0)": ScoringRules.standard(),
    "TE Premium (1.5)": ScoringRules.te_premium(0.5)
}

roster_settings = RosterSettings(qb=1, rb=2, wr=2, te=1, flex=2, bench=6)
settings = LeagueSettings(
    name="Streamlit Simulator League",
    teams_count=league_size,
    scoring=scoring_map[scoring_type],
    roster=roster_settings,
    faab_budget=100
)

# App Navigation tabs
tab_sandbox, tab_draft, tab_training, tab_stats = st.tabs([
    "🏆 Sandbox Season Simulator",
    "🎯 Live AI Draft Room",
    "🧠 AI Training Grounds",
    "📊 Player Database"
])

# Initialize session states
if "sandbox" not in st.session_state:
    st.session_state.sandbox = None
if "draft_room" not in st.session_state:
    st.session_state.draft_room = None

# ==============================================================================
# TAB 1: Sandbox Season Simulator
# ==============================================================================
with tab_sandbox:
    st.header("🏆 League Sandbox Season Playback")
    st.write("Run a full week-by-week fantasy football season and watch 8 distinct AI personas draft, trade, bid FAAB on waivers, and optimize starting lineups.")
    
    sb = st.session_state.sandbox
    col_act, col_sett = st.columns([1, 3])
    
    with col_act:
        # Load available personas (defaults + custom models)
        default_personas = [
            "balanced", "free_agent_demon", "trade_demon", "matchup_all_star",
            "conservative", "zero_rb", "hero_rb", "high_risk", "late_round_qb", "robust_rb"
        ]
        saved_models = db.get_all_trained_models()
        available_styles = default_personas + saved_models

        if st.session_state.sandbox is None:
            st.write("### 🛠️ Customize Sandbox League")
            st.info("Set up team names and playstyle personas. When ready, click 'Draft Teams & Start Season' to begin.")
            
            team_configs = []
            for i in range(league_size):
                col_name, col_style = st.columns([2, 1])
                with col_name:
                    t_name = st.text_input(f"Team {i+1} Name", f"Team {i+1} (AI)" if i > 0 else "My Franchise", key=f"tname_key_{i}")
                with col_style:
                    # Select playstyle
                    default_style_idx = i % len(default_personas)
                    t_style = st.selectbox(
                        f"Owner {i+1} Playstyle", 
                        available_styles, 
                        index=default_style_idx if default_style_idx < len(available_styles) else 0,
                        key=f"tstyle_key_{i}"
                    )
                team_configs.append((t_name, t_style))
                
            btn_start_sim = st.button("🚀 Draft Teams & Start Season", use_container_width=True)
            
            if btn_start_sim:
                # Create sandbox
                sb = LeagueSandbox(settings, year=sim_year)
                
                teams = []
                for i, (t_name, t_style) in enumerate(team_configs):
                    teams.append(Team(
                        id=f"team_{i+1}",
                        name=t_name,
                        owner_persona=t_style,
                        roster=Roster(),
                        faab_balance=100
                    ))
                sb.initialize_league(teams)
                
                # Run draft
                sb.start_draft()
                sb.auto_draft_fill()
                
                st.session_state.sandbox = sb
                st.session_state.sb_week = 1
                st.success("Draft completed and sandbox initialized!")
                st.rerun()
        else:
            sb = st.session_state.sandbox
            st.write("### Simulation Controls")
            st.write(f"**Current Week:** {sb.current_week if sb.current_week <= 14 else 'Complete'}")
            
            if sb.current_week <= 14:
                btn_next_week = st.button("🏈 Play Week Matchups", use_container_width=True)
                if btn_next_week:
                    # Lineup optimization, waivers, trades
                    all_players = get_cached_players()
                    projs = get_cached_projs(sim_year, sb.current_week)
                    
                    # Instantiate agents
                    agents = {t.id: get_agent_by_persona(t.owner_persona, t.id, settings) for t in sb.teams.values()}
                    
                    # Lineups
                    for t_id, team in sb.teams.items():
                        agent = agents[t_id]
                        starters, bench = agent.optimize_weekly_lineup(team.roster, projs)
                        sb.set_lineup(t_id, starters, bench, [])
                        
                    # Waivers
                    free_agents = [p for p in all_players if p.id not in sb.draft_state.drafted_player_ids]
                    all_claims = []
                    for t_id, team in sb.teams.items():
                        agent = agents[t_id]
                        claims = agent.get_waiver_claims(team, free_agents[:50], projs, current_week=sb.current_week)
                        all_claims.extend(claims)
                    sb.process_waiver_claims(all_claims)
                    
                    # Trades
                    trade_proposals = []
                    for t_id, team in sb.teams.items():
                        agent = agents[t_id]
                        if hasattr(agent, "generate_trade_proposals"):
                            proposals = agent.generate_trade_proposals(team, list(sb.teams.values()), projs)
                            trade_proposals.extend(proposals)
                    for proposal in trade_proposals:
                        recv_agent = agents[proposal.receiver_team_id]
                        recv_team = sb.teams[proposal.receiver_team_id]
                        if recv_agent.evaluate_trade_proposal(recv_team, proposal, projs):
                            sb.execute_trade(proposal)
                            
                    # Matchups
                    sb.simulate_week()
                    st.rerun()
            else:
                st.success("🎉 The season is complete!")
                
            btn_reset_sim = st.button("🔄 Reset Season & Reconfigure Teams", use_container_width=True)
            if btn_reset_sim:
                st.session_state.sandbox = None
                st.rerun()

    with col_sett:
        if sb:
            tab_standings, tab_rosters, tab_transactions = st.tabs(["📊 Standings", "🏃 Rosters", "🗞️ Waiver/Trade Logs"])
            
            with tab_standings:
                st.write("### Current Standings")
                standings = sb.get_standings()
                standings_data = []
                for idx, team in enumerate(standings):
                    standings_data.append({
                        "Rank": idx + 1,
                        "Franchise": team.name,
                        "Persona Style": team.owner_persona.replace("_", " ").title(),
                        "Record": team.record_str,
                        "Points For": team.points_for,
                        "Points Against": team.points_against,
                        "FAAB Balance": f"${team.faab_balance}"
                    })
                st.table(pd.DataFrame(standings_data).set_index("Rank"))
                
                # Matchups
                st.write("### Matchup Scores")
                week_to_show = max(1, sb.current_week - 1)
                st.markdown(f"**Week {week_to_show} Matchups:**")
                for matchup in sb.schedule.get(week_to_show, []):
                    team_a = sb.teams[matchup.team_a_id].name
                    team_b = sb.teams[matchup.team_b_id].name
                    st.write(f"- {team_a} (**{matchup.team_a_score}**) vs {team_b} (**{matchup.team_b_score}**)")
                    
            with tab_rosters:
                team_to_view = st.selectbox("Select Team to Inspect", list(sb.teams.keys()), format_func=lambda x: sb.teams[x].name)
                t = sb.teams[team_to_view]
                st.write(f"### Roster for {t.name}")
                st.write(f"**Owner Persona:** {t.owner_persona.replace('_', ' ').title()}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("#### Starters")
                    for p in t.roster.starters:
                        pos = p.position.value if hasattr(p.position, "value") else str(p.position)
                        st.markdown(f"<span class='badge badge-{pos.lower()}'>{pos}</span> **{p.name}** ({p.nfl_team})", unsafe_allow_html=True)
                with col2:
                    st.write("#### Bench")
                    for p in t.roster.bench:
                        pos = p.position.value if hasattr(p.position, "value") else str(p.position)
                        st.markdown(f"<span class='badge badge-{pos.lower()}'>{pos}</span> **{p.name}** ({p.nfl_team})", unsafe_allow_html=True)
                        
            with tab_transactions:
                st.write("### Transaction Log")
                if not sb.transaction_history:
                    st.info("No waiver claims, drafts, or trades processed yet.")
                else:
                    for txn in reversed(sb.transaction_history):
                        st.markdown(f"**[{txn['type']} - Week {txn['week']}]** {txn['team_name']}: {txn['player_name']} - *{txn['details']}*")

        else:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 3rem;">
                <h2 style="margin: 0; color: #d76d77;">🏈 Sandbox League is Ready!</h2>
                <p style="opacity: 0.8; font-size: 1.1rem; margin-top: 1rem;">
                    Configure your team names and AI personas on the left side of the dashboard, 
                    then click the draft button to build your rosters and unlock the standings, rosters, and trade logs tabs.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: Live AI Draft Room
# ==============================================================================
with tab_draft:
    st.header("🎯 Live AI Draft Room")
    st.write("Create a league, join the draft room as the human owner, and draft your roster in real-time. Watch the remaining teams employ distinct drafting philosophies (Zero-RB, Hero-RB, Risk-Averse).")
    
    btn_start_draft = st.button("🏁 Start Draft Session")
    
    if btn_start_draft or st.session_state.draft_room is None:
        # Create draft room sandbox
        ds = LeagueSandbox(settings, year=sim_year)
        
        # Build teams: Team 1 is human, rest are AI
        teams = [Team(id="team_1", name="My Franchise (Human)", owner_persona="human", roster=Roster(), faab_balance=100)]
        personas = ["zero_rb", "hero_rb", "high_risk", "conservative", "free_agent_demon", "trade_demon", "balanced", "late_round_qb", "robust_rb"]
        # Prioritize custom saved trained models as draft opponents
        saved_models = db.get_all_trained_models()
        opponent_pool = saved_models + personas
        
        # fill
        while len(teams) < league_size:
            p = opponent_pool[(len(teams) - 1) % len(opponent_pool)]
            teams.append(Team(
                id=f"team_{len(teams)+1}",
                name=f"Team {p.replace('_', ' ').title()} (AI)",
                owner_persona=p,
                roster=Roster()
            ))
            
        ds.initialize_league(teams)
        # Fixed draft order: Human drafts first
        ds.start_draft()
        # Ensure draft order starts with human for first pick
        ds.draft_state.draft_order = [t.id for t in teams]
        st.session_state.draft_room = ds
        
    ds = st.session_state.draft_room
    
    if ds and ds.draft_state:
        dstate = ds.draft_state
        total_picks = dstate.rounds * len(ds.teams)
        
        if len(dstate.picks) >= total_picks:
            st.success("🎉 The draft is complete!")
        else:
            current_team_id = dstate.get_current_team_id()
            current_team = ds.teams[current_team_id]
            
            st.write(f"### Current Turn: **{current_team.name}**")
            st.write(f"Round **{dstate.current_round}**, Pick **{len(dstate.picks)+1}** of {total_picks}")
            
            # Load projections for draft decisions
            projs = get_cached_projs(sim_year, 1)
            all_players = get_cached_players()
            undrafted = [p for p in all_players if p.id not in dstate.drafted_player_ids]
            
            # Sort undrafted by projection
            undrafted_sorted = sorted(undrafted, key=lambda p: projs.get(p.id, 0.0), reverse=True)
            
            if current_team.owner_persona == "human":
                st.write("👉 **Your turn to pick! Select a player below:**")
                
                # Allow search/filter by position
                filter_pos = st.selectbox("Filter Position", ["ALL", "QB", "RB", "WR", "TE", "K", "DST"])
                filtered_players = undrafted_sorted
                if filter_pos != "ALL":
                    filtered_players = [p for p in undrafted_sorted if p.position.value if hasattr(p.position, "value") and p.position.value == filter_pos or str(p.position) == filter_pos]
                    
                selected_draft_player = st.selectbox(
                    "Select Player",
                    filtered_players[:100],
                    format_func=lambda p: f"{p.name} ({p.position.value if hasattr(p.position, 'value') else p.position} - {p.nfl_team}) - Proj: {projs.get(p.id, 0.0)} pts"
                )
                
                btn_draft = st.button("🏈 Submit Draft Selection")
                if btn_draft:
                    ds.execute_draft_pick(selected_draft_player)
                    st.rerun()
            else:
                st.write("🤖 AI is deciding...")
                btn_ai_pick = st.button("▶️ Let AI Make Pick", use_container_width=True)
                if btn_ai_pick:
                    # Instantiate the agent
                    agent = get_agent_by_persona(current_team.owner_persona, current_team_id, settings)
                    selected_player = agent.draft_pick(dstate, undrafted, projs)
                    ds.execute_draft_pick(selected_player)
                    st.rerun()
                    
        # Show Draft Board
        col_board, col_my_team = st.columns([2, 1])
        with col_board:
            st.write("### Draft Board (Recent Picks)")
            if not dstate.picks:
                st.info("Draft board is empty.")
            else:
                board_df = []
                for p in reversed(dstate.picks):
                    player = p["player"]
                    board_df.append({
                        "Pick": p["pick_number"],
                        "Round": p["round"],
                        "Franchise": ds.teams[p["team_id"]].name,
                        "Player": player.name,
                        "Pos": player.position.value if hasattr(player.position, "value") else player.position,
                        "NFL Team": player.nfl_team
                    })
                st.dataframe(pd.DataFrame(board_df).set_index("Pick").head(20))
                
        with col_my_team:
            st.write("### My Franchise Roster")
            my_team = ds.teams["team_1"]
            for idx, p in enumerate(my_team.roster.all_players()):
                pos = p.position.value if hasattr(p.position, "value") else str(p.position)
                st.markdown(f"{idx+1}. <span class='badge badge-{pos.lower()}'>{pos}</span> **{p.name}** ({p.nfl_team})", unsafe_allow_html=True)

# ==============================================================================
# TAB 3: AI Training Grounds
# ==============================================================================
# ==============================================================================
# TAB 3: AI Training Grounds
# ==============================================================================
with tab_training:
    st.header("🧠 AI Training Grounds")
    st.write("Train and perfect custom fantasy football playstyles or evolve a Super Expert hybrid manager. Training runs seasons in parallel in the background, shuffling draft positions and reference years (2022–2025) while keeping specific strategy constraints intact.")
    
    col_ga_controls, col_ga_results = st.columns([1, 2])
    
    # Check if a background training run is active by reading the progress file
    PROGRESS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "training_progress.json")
    
    progress_data = None
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                progress_data = json.load(f)
        except Exception:
            pass
            
    is_training_active = bool(progress_data and progress_data.get("status") == "running")
    
    with col_ga_controls:
        st.write("### New Training Settings")
        
        # 10 playstyles list + hybrid
        styles_list = [
            "hybrid", "balanced", "free_agent_demon", "trade_demon", "matchup_all_star",
            "conservative", "zero_rb", "hero_rb", "high_risk", "late_round_qb", "robust_rb"
        ]
        
        train_style = st.selectbox(
            "Target playstyle to train/perfect",
            styles_list,
            format_func=lambda x: "Super Expert (Hybrid)" if x == "hybrid" else f"Perfecting: {x.replace('_', ' ').title()}",
            disabled=is_training_active
        )
        
        ga_generations = st.slider("Generations (Training epochs)", 1, 50, 5, disabled=is_training_active)
        ga_pop_size = st.selectbox("Population Size (Multiples of 10)", [10, 20, 30, 40], index=1, disabled=is_training_active)
        ga_seasons = st.slider("Simulated Seasons per Eval (Avg sample size)", 1, 20, 5, disabled=is_training_active)
        
        btn_train_ga = st.button("🧬 Launch Background Training", use_container_width=True, disabled=is_training_active)
        
        if btn_train_ga:
            import subprocess
            # Launch train_daemon.py in background
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai", "train_daemon.py"),
                "--playstyle", train_style,
                "--generations", str(ga_generations),
                "--seasons", str(ga_seasons),
                "--pop-size", str(ga_pop_size)
            ]
            
            # Start process without waiting (background daemon)
            subprocess.Popen(cmd)
            st.success("🧬 Evolutionary training launched in the background! Progress will display on the right.")
            st.rerun()
            
        if is_training_active:
            st.info("⏳ A training run is currently active. Settings are locked.")
            btn_kill_train = st.button("🛑 Force Stop Training Run", use_container_width=True)
            if btn_kill_train:
                with open(PROGRESS_FILE, "w") as f:
                    json.dump({"status": "failed", "playstyle": progress_data.get("playstyle"), "current_generation": 0, "total_generations": 0, "error": "Forced stop by user"}, f)
                st.warning("Training stopped.")
                st.rerun()

    with col_ga_results:
        st.write("### Training Status & Results")
        
        if not progress_data:
            st.info("No training history found. Launch a training run on the left.")
        else:
            status = progress_data.get("status")
            style = progress_data.get("playstyle", "unknown")
            gen = progress_data.get("current_generation", 0)
            total_gens = progress_data.get("total_generations", 1)
            
            if status == "running":
                st.markdown(f"#### ⚡ Active Job: Perfecting **{style.replace('_', ' ').title()}**")
                # Progress Bar
                progress_val = min(1.0, max(0.0, gen / total_gens))
                st.progress(progress_val)
                st.write(f"Evaluating generation **{gen} of {total_gens}**...")
                st.write(f"Current Generation Top Fitness: **{progress_data.get('top_fitness', 0.0):.2f}**")
                st.write(f"Current Generation Avg Fitness: **{progress_data.get('avg_fitness', 0.0):.2f}**")
                
                # Dynamic update page refresh button
                st.button("🔄 Refresh Progress Output", use_container_width=True)
                
            elif status == "completed":
                st.success(f"🎉 Training Completed successfully! Perfected model for **{style.replace('_', ' ').title()}** is saved permanently to SQLite.")
                
                best_model = progress_data.get("best_model")
                if best_model:
                    st.write("#### Evolved Parameter Configuration:")
                    params_df = pd.DataFrame([
                        {"Parameter Field": k, "Evolved Weight": f"{v:.4f}" if isinstance(v, float) else str(v)}
                        for k, v in best_model.items()
                    ])
                    st.dataframe(params_df, use_container_width=True)
                    
            elif status == "failed":
                st.error("❌ The training run failed or was stopped.")
                st.write(f"Error Log: {progress_data.get('error')}")
                
            if status == "running" and progress_data.get("best_model"):
                with st.expander("🔍 View Best Model Parameters So Far"):
                    best_model = progress_data.get("best_model")
                    params_df = pd.DataFrame([
                        {"Parameter Field": k, "Evolved Weight": f"{v:.4f}" if isinstance(v, float) else str(v)}
                        for k, v in best_model.items()
                    ])
                    st.dataframe(params_df, use_container_width=True)

# ==============================================================================
# TAB 4: Player Database & Stats
# ==============================================================================
with tab_stats:
    st.header("📊 Player Database & Stats")
    st.write("Browse through the NFL players saved in the local cache, view their core status, positions, and weekly stats for the 2024 season.")
    
    players = get_cached_players()
    
    if not players:
        st.warning("Database contains no players. Run a simulation first to fetch rosters.")
    else:
        st.write(f"Total Cached Players: **{len(players)}**")
        
        col_search, col_pos = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("🔍 Search Player by Name", "")
        with col_pos:
            pos_filter = st.selectbox("Position Group", ["ALL", "QB", "RB", "WR", "TE", "K", "DST"])
            
        # Filter
        filtered_p = players
        if search_query:
            filtered_p = [p for p in filtered_p if search_query.lower() in p.name.lower()]
        if pos_filter != "ALL":
            filtered_p = [p for p in filtered_p if p.position.value if hasattr(p.position, "value") and p.position.value == pos_filter or str(p.position) == pos_filter]
            
        # Display as a dataframe
        players_data = []
        for p in filtered_p[:100]: # limit to top 100 for performance
            players_data.append({
                "Player Name": p.name,
                "Position": p.position.value if hasattr(p.position, "value") else p.position,
                "NFL Team": p.nfl_team,
                "Age": p.age if p.age else "N/A",
                "Exp": p.experience if p.experience else 0,
                "Status": p.status
            })
            
        st.dataframe(pd.DataFrame(players_data), use_container_width=True)
