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
from src.ai.base_agent import BaseAgent
from src.ai.best_agent import ProAIEngine, get_pro_ai_agent
from src.espn.espn_client import ESPNClient
from src.espn.advisor import ESPNStrategyAdvisor
from src.engine.inseason_app import InSeasonLeagueEngine
from src.data import db

# Set page config for SEO and layout
st.set_page_config(
    page_title="Antigravity Fantasy Football AI Simulator",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Ultra-Compact Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Ultra-Compact Layout for Clean Screen Fit */
    .block-container {
        padding-top: 0.3rem !important;
        padding-bottom: 0.3rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 98% !important;
    }
    
    .nav-header {
        background: linear-gradient(90deg, #1e1b4b, #312e81, #4338ca);
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.15em 0.45em;
        font-size: 70%;
        font-weight: 700;
        border-radius: 0.3rem;
        color: white;
        margin-right: 0.3rem;
    }
    
    .badge-qb { background-color: #2e62ff; }
    .badge-rb { background-color: #1aa160; }
    .badge-wr { background-color: #8c2eff; }
    .badge-te { background-color: #ff7700; }
    .badge-k { background-color: #ff3377; }
    .badge-dst { background-color: #6c757d; }

    /* Compact metric spacing */
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Main compact navbar title
st.markdown("""
<div class="nav-header">
    <div style="font-weight: 800; font-size: 1.3rem;">🏈 Antigravity Fantasy Football AI Platform</div>
    <div style="font-size: 0.85rem; opacity: 0.85;">⚡ Powered by Pro AI Engine</div>
</div>
""", unsafe_allow_html=True)

# AI Player News & Predictions Ticker
try:
    from src.ai.predictions import AINewsPredictionEngine
    news_engine = AINewsPredictionEngine()
    news_items = news_engine.get_all_news_highlights()
    with st.expander("📰 **AI Player News, Team Roster Updates & 2026 Forecasts**", expanded=False):
        n_cols = st.columns(3)
        for idx, item in enumerate(news_items[:6]):
            with n_cols[idx % 3]:
                st.markdown(f"**{item['player_name']}** (`{item['team']}`)\n\n{item['news']}\n\n*🔮 AI Forecast:* {item['forecast']}")
except Exception:
    pass

# Helper to fetch players and projections once
@st.cache_data
def get_cached_players():
    db.init_db()
    return db.get_all_players()

@st.cache_data
def get_cached_projs(year, week):
    return {p.player_id: p.projected_points for p in db.get_weekly_projections(year, week)}

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

# App Navigation tabs (Training Grounds removed as requested)
tab_espn, tab_draft, tab_inseason, tab_sandbox, tab_stats = st.tabs([
    "⚡ ESPN AI League Advisor",
    "🎯 Live 10-AI Spectator Draft",
    "🎮 Interactive In-Season App",
    "🏆 Sandbox Season Simulator",
    "📊 Player Database"
])

# Initialize session states
if "sandbox" not in st.session_state:
    st.session_state.sandbox = None
if "draft_room" not in st.session_state:
    st.session_state.draft_room = None
if "espn_client" not in st.session_state:
    st.session_state.espn_client = None
if "inseason_engine" not in st.session_state:
    st.session_state.inseason_engine = None

# ==============================================================================
# TAB 1: ESPN AI League Advisor
# ==============================================================================
with tab_espn:
    st.subheader("⚡ ESPN AI League Advisor")
    col_espn_inputs, col_espn_output = st.columns([1, 2])
    
    with col_espn_inputs:
        st.write("### 🔗 ESPN League Connection")
        espn_id_input = st.number_input("ESPN League ID (Enter '0' for Demo League)", min_value=0, value=0, step=1)
        espn_year_input = st.selectbox("Season Year", [2025, 2024, 2023, 2022], index=0, key="espn_year_sel")
        
        with st.expander("🔑 Private League Cookies (Optional)"):
            espn_s2_input = st.text_input("espn_s2 Cookie", "", type="password")
            swid_input = st.text_input("swid Cookie", "")
            
        btn_connect_espn = st.button("⚡ Connect & Generate AI Advice")
        
        if btn_connect_espn:
            client = ESPNClient(
                league_id=int(espn_id_input),
                year=int(espn_year_input),
                espn_s2=espn_s2_input.strip(),
                swid=swid_input.strip()
            )
            with st.spinner("Connecting to ESPN..."):
                client.connect()
                st.session_state.espn_client = client
                st.success("Connected to ESPN League!" if not client.is_mock else "Connected using Demo mode!")
                st.rerun()

    with col_espn_output:
        if st.session_state.espn_client is None:
            st.info("👈 Enter your ESPN League ID on the left and click 'Connect & Generate AI Advice' to view strategy recommendations.")
        else:
            client = st.session_state.espn_client
            agent = ProAIEngine("user_espn_agent", settings)
            advisor = ESPNStrategyAdvisor(agent)
            
            teams = client.get_teams_internal()
            if not teams:
                st.error("No teams found in ESPN league.")
            else:
                user_team = teams[0]
                st.markdown(f"### 🏈 **{user_team.name}**")
                st.caption(f"Driven by **Pro AI Engine** | Record: **{user_team.record_str}** | FAAB: **${user_team.faab_balance}**")
                
                projs = get_cached_projs(sim_year, 1)
                free_agents = client.get_free_agents_internal(week=1)
                
                adv_tab_start, adv_tab_waiver, adv_tab_trade, adv_tab_draft = st.tabs([
                    "📌 Start / Sit",
                    "🌊 Waiver Wire",
                    "🤝 Trade Offers",
                    "🎯 Draft Helper"
                ])
                
                with adv_tab_start:
                    res = advisor.analyze_start_sit(user_team, projs)
                    c_cur, c_opt, c_gain = st.columns(3)
                    c_cur.metric("Current Projected", f"{res['current_projected']} pts")
                    c_opt.metric("Optimal Lineup", f"{res['optimal_projected']} pts")
                    c_gain.metric("Potential Gain", f"+{res['potential_gain']} pts")
                    
                    if not res["recommended_swaps"]:
                        st.success("✅ Current lineup is optimal!")
                    else:
                        for swap in res["recommended_swaps"]:
                            p_start = swap["start_player"]
                            p_bench = swap["bench_player"]
                            st.info(f"🟢 **START** {p_start.name} ({swap['start_proj']} pts) ➔ 🔴 **BENCH** {p_bench.name} ({swap['bench_proj']} pts) | **+{swap['point_gain']} pts**")
                            
                with adv_tab_waiver:
                    waiver_recs = advisor.analyze_waivers(user_team, free_agents, projs, week=1)
                    if not waiver_recs:
                        st.info("No waiver additions recommended.")
                    else:
                        for rec in waiver_recs:
                            st.success(f"➕ **Add:** {rec['add_player'].name} ({rec['add_proj']} pts/wk) | ➖ **Drop:** {rec['drop_player'].name if rec['drop_player'] else 'None'} | 💰 **Bid:** **${rec['bid_amount']}**")

                with adv_tab_trade:
                    trades = advisor.analyze_trades(user_team, teams, projs)
                    if not trades:
                        st.info("No trade offers meet the AI score threshold.")
                    else:
                        for tr in trades:
                            st.warning(f"🤝 **Trade Offer ({tr['target_team']})**: {tr['summary']}")

                with adv_tab_draft:
                    draft_recs = advisor.analyze_draft_picks(free_agents, projs, None, top_n=5)
                    draft_df = [{"Player": r["player"].name, "Pos": r["position"], "Team": r["player"].nfl_team, "Proj": r["projected_points"], "VORP": r["vorp"]} for r in draft_recs]
                    st.dataframe(pd.DataFrame(draft_df), height=200)

# ==============================================================================
# TAB 2: Live 10-AI Spectator Draft
# ==============================================================================
with tab_draft:
    st.subheader("🎯 Live 10-AI Spectator Draft Room & Season Simulator")
    
    if "spectator_engine" not in st.session_state:
        st.session_state.spectator_engine = None
        
    col_spec_ctrl, col_spec_view = st.columns([1, 2])
    
    with col_spec_ctrl:
        btn_init_spec = st.button("🚀 Initialize 10 Pro-AI Draft")
        if btn_init_spec or st.session_state.spectator_engine is None:
            from src.engine.spectator_draft import SpectatorDraftEngine
            s_engine = SpectatorDraftEngine(settings, year=sim_year)
            s_engine.initialize_draft()
            st.session_state.spectator_engine = s_engine
            st.rerun()
            
        s_engine = st.session_state.spectator_engine
        
        if s_engine:
            ds = s_engine.sandbox.draft_state
            total_picks = s_engine.settings.roster.total_roster_spots() * len(s_engine.sandbox.teams)
            st.write(f"**Picks Completed:** {len(ds.picks)} / {total_picks}")
            st.progress(len(ds.picks) / total_picks)
            
            if not s_engine.is_complete:
                c1, c2 = st.columns(2)
                with c1:
                    btn_step = st.button("▶️ Next Pick")
                    if btn_step:
                        s_engine.step_next_pick()
                        st.rerun()
                with c2:
                    btn_fast = st.button("⏭️ Complete Draft")
                    if btn_fast:
                        s_engine.auto_complete_draft()
                        st.rerun()
            else:
                st.success("🎉 Draft Complete!")
                btn_sim_season = st.button("🚀 Play Full Season & Playoffs")
                if btn_sim_season:
                    for wk in range(1, 15):
                        weekly_projs = get_cached_projs(sim_year, wk)
                        for t_id, team in s_engine.sandbox.teams.items():
                            agent = s_engine.agents[t_id]
                            starters, bench = agent.optimize_weekly_lineup(team.roster, weekly_projs)
                            s_engine.sandbox.set_lineup(t_id, starters, bench, [])
                        s_engine.sandbox.simulate_week()
                    st.session_state.spectator_completed = True
                    st.rerun()
                    
    with col_spec_view:
        if s_engine:
            if not s_engine.pick_logs:
                st.info("Draft initialized with Pro AI Engines. Click '▶️ Next Pick' or '⏭️ Complete Draft'.")
            else:
                latest_pick = s_engine.pick_logs[-1]
                p = latest_pick["player"]
                st.markdown(f"""
                <div class="card">
                    <b>Round {latest_pick['round']}, Pick {latest_pick['pick_number']}: {latest_pick['team_name']}</b><br/>
                    Selected <strong>{p.name}</strong> ({latest_pick['position']} - {p.nfl_team}) &nbsp;|&nbsp; <em>{latest_pick['reasoning']}</em>
                </div>
                """, unsafe_allow_html=True)
                
                log_df = [{"Pick": log["pick_number"], "Round": log["round"], "Franchise": log["team_name"], "Player": log["player"].name, "Pos": log["position"], "VORP": f"+{log['vorp']}"} for log in reversed(s_engine.pick_logs)]
                st.dataframe(pd.DataFrame(log_df).set_index("Pick"), height=220)

            if getattr(st.session_state, "spectator_completed", False):
                standings = s_engine.sandbox.get_standings()
                st.success(f"🏆 Champion: **{standings[0].name}** ({standings[0].record_str}) - {standings[0].points_for} pts")
                s_data = [{"Rank": idx + 1, "Franchise": t.name, "Record": t.record_str, "Points For": t.points_for, "FAAB": f"${t.faab_balance}"} for idx, t in enumerate(standings)]
                st.table(pd.DataFrame(s_data).set_index("Rank"))

            # Team Roster Inspector Widget
            st.write("---")
            st.markdown("#### 📋 View Franchise Roster & Drafted Players")
            all_teams = list(s_engine.sandbox.teams.values())
            selected_team = st.selectbox("Select Franchise to Inspect Roster:", all_teams, format_func=lambda t: t.name, key="spec_roster_select")
            if selected_team:
                r_players = selected_team.roster.all_players()
                if not r_players:
                    st.caption("No players drafted yet for this team.")
                else:
                    projs = s_engine.projections
                    r_data = []
                    for p in r_players:
                        pos_str = p.position.value if hasattr(p.position, "value") else str(p.position)
                        r_data.append({
                            "Position": pos_str,
                            "Player Name": p.name,
                            "NFL Team": p.nfl_team,
                            "Projected Pts/Wk": projs.get(p.id, 0.0)
                        })
                    st.dataframe(pd.DataFrame(r_data), height=180)


# ==============================================================================
# TAB 3: Interactive In-Season App
# ==============================================================================
with tab_inseason:
    st.subheader("🎮 Interactive In-Season League App")
    
    if st.session_state.inseason_engine is None:
        col_name, col_btn = st.columns([3, 1])
        with col_name:
            user_fname = st.text_input("Franchise Name", "My Championship Team", key="inseason_fname_input")
        with col_btn:
            btn_start_inseason = st.button("🏁 Start Season", key="btn_start_inseason_key")
            
        if btn_start_inseason:
            engine = InSeasonLeagueEngine(settings, year=sim_year)
            with st.spinner("Initializing league against 9 Pro AI Opponents..."):
                engine.initialize_season(user_team_name=user_fname)
                st.session_state.inseason_engine = engine
                st.rerun()
    else:
        engine = st.session_state.inseason_engine
        user_team = engine.user_team
        wk = engine.current_week
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Week", f"Week {wk if wk <= 14 else 'Complete'}")
        m2.metric("Record", user_team.record_str)
        m3.metric("FAAB", f"${user_team.faab_balance}")
        m4.metric("AI Opponents", "9 Pro AI Engines")
        
        if engine.ai_reaction_logs:
            for log in engine.ai_reaction_logs:
                st.info(f"⚡ **AI Reaction**: {log}")
                
        in_tab_roster, in_tab_waiver, in_tab_sync, in_tab_standings = st.tabs([
            "🏃 Lineup Editor",
            "🌊 Waiver Console",
            "🏈 Play Weekend",
            "📊 Standings"
        ])
        
        projs = get_cached_projs(sim_year, wk if wk <= 14 else 14)
        
        with in_tab_roster:
            c_starters, c_bench = st.columns(2)
            with c_starters:
                starter_to_bench = st.selectbox("Starter to Bench", user_team.roster.starters, format_func=lambda p: f"{p.name} ({p.position.value if hasattr(p.position, 'value') else p.position}) - {projs.get(p.id, 0.0)} pts", key="s_swap")
            with c_bench:
                bench_to_start = st.selectbox("Bench to Starter", user_team.roster.bench, format_func=lambda p: f"{p.name} ({p.position.value if hasattr(p.position, 'value') else p.position}) - {projs.get(p.id, 0.0)} pts", key="b_swap")
                
            btn_swap = st.button("🔄 Execute Swap & Trigger AI Reactions")
            if btn_swap and starter_to_bench and bench_to_start:
                new_starters = [p for p in user_team.roster.starters if p.id != starter_to_bench.id] + [bench_to_start]
                new_bench = [p for p in user_team.roster.bench if p.id != bench_to_start.id] + [starter_to_bench]
                engine.update_user_lineup(new_starters, new_bench)
                st.rerun()

        with in_tab_waiver:
            all_players = get_cached_players()
            drafted_ids = {p.id for t in engine.sandbox.teams.values() for p in t.roster.all_players()}
            free_agents = [p for p in all_players if p.id not in drafted_ids]
            
            col_fa_add, col_fa_drop, col_fa_bid = st.columns([2, 2, 1])
            with col_fa_add:
                fa_add = st.selectbox("Add Free Agent", free_agents[:50], format_func=lambda p: f"{p.name} ({p.position.value if hasattr(p.position, 'value') else p.position}) - {projs.get(p.id, 0.0)} pts", key="in_fa_add")
            with col_fa_drop:
                fa_drop = st.selectbox("Drop Player", user_team.roster.bench, format_func=lambda p: f"{p.name} ({p.position.value if hasattr(p.position, 'value') else p.position})", key="in_fa_drop")
            with col_fa_bid:
                fa_bid = st.number_input("FAAB Bid ($)", min_value=0, max_value=user_team.faab_balance, value=5, step=1, key="in_fa_bid")
                
            btn_claim = st.button("💰 Submit Claim & Alert Pro AIs")
            if btn_claim and fa_add:
                engine.submit_user_waiver_claim(fa_add, fa_drop, int(fa_bid))
                st.rerun()

        with in_tab_sync:
            if wk > 14:
                st.success("🎉 Season completed!")
            else:
                btn_play_weekend = st.button(f"🏈 Play Week {wk} Weekend Games")
                if btn_play_weekend:
                    engine.simulate_weekend_games()
                    st.rerun()

        with in_tab_standings:
            standings = engine.sandbox.get_standings()
            s_data = [{"Rank": idx + 1, "Franchise": t.name, "Record": t.record_str, "Points For": t.points_for, "FAAB": f"${t.faab_balance}"} for idx, t in enumerate(standings)]
            st.table(pd.DataFrame(s_data).set_index("Rank"))

# ==============================================================================
# TAB 4: Sandbox Season Simulator
# ==============================================================================
with tab_sandbox:
    st.subheader("🏆 League Sandbox Season Playback")
    sb = st.session_state.sandbox
    col_act, col_sett = st.columns([1, 2])
    
    with col_act:
        if st.session_state.sandbox is None:
            btn_start_sim = st.button("🚀 Draft 10 Pro AI Teams & Start Season")
            if btn_start_sim:
                sb = LeagueSandbox(settings, year=sim_year)
                teams = [Team(id=f"team_{i+1}", name=f"Pro AI Team {i+1}", owner_persona="Pro AI Engine", roster=Roster(), faab_balance=100) for i in range(league_size)]
                sb.initialize_league(teams)
                sb.start_draft()
                sb.auto_draft_fill()
                st.session_state.sandbox = sb
                st.rerun()
        else:
            sb = st.session_state.sandbox
            st.write(f"**Current Week:** {sb.current_week if sb.current_week <= 14 else 'Complete'}")
            if sb.current_week <= 14:
                btn_next_week = st.button("🏈 Play Week Matchups")
                if btn_next_week:
                    projs = get_cached_projs(sim_year, sb.current_week)
                    agents = {t.id: ProAIEngine(t.id, settings) for t in sb.teams.values()}
                    for t_id, team in sb.teams.items():
                        starters, bench = agents[t_id].optimize_weekly_lineup(team.roster, projs)
                        sb.set_lineup(t_id, starters, bench, [])
                    sb.simulate_week()
                    st.rerun()
            else:
                st.success("🎉 Season complete!")
            btn_reset_sim = st.button("🔄 Reset Sandbox")
            if btn_reset_sim:
                st.session_state.sandbox = None
                st.rerun()

    with col_sett:
        if sb:
            standings = sb.get_standings()
            s_data = [{"Rank": idx + 1, "Franchise": t.name, "Record": t.record_str, "Points For": t.points_for, "FAAB": f"${t.faab_balance}"} for idx, t in enumerate(standings)]
            st.table(pd.DataFrame(s_data).set_index("Rank"))

# ==============================================================================
# TAB 5: Player Database & Stats
# ==============================================================================
with tab_stats:
    st.subheader("📊 Player Database")
    players = get_cached_players()
    if players:
        col_search, col_pos = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("🔍 Search Player", "")
        with col_pos:
            pos_filter = st.selectbox("Position", ["ALL", "QB", "RB", "WR", "TE", "K", "DST"])
            
        filtered_p = players
        if search_query:
            filtered_p = [p for p in filtered_p if search_query.lower() in p.name.lower()]
        if pos_filter != "ALL":
            filtered_p = [p for p in filtered_p if p.position.value if hasattr(p.position, "value") and p.position.value == pos_filter or str(p.position) == pos_filter]
            
        players_data = [{"Name": p.name, "Position": p.position.value if hasattr(p.position, "value") else p.position, "Team": p.nfl_team, "Age": p.age if p.age else "N/A", "Exp": p.experience if p.experience else 0} for p in filtered_p[:100]]
        st.dataframe(pd.DataFrame(players_data), height=240)
