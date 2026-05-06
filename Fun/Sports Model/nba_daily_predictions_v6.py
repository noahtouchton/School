import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import datetime, timezone

# --- CONFIGURATION ---
API_KEY = '2c53c9affcb2d9c702b0538ba851f206' # <--- PASTE YOUR API KEY HERE
RAW_DATA_FILE = 'Fun/Sports Model/nba_advanced_master.csv'

TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

def get_current_advanced_stats(df):
    """
    Calculates the CURRENT advanced momentum (EWMA) and Days Rest.
    """
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])
    
    current_stats = {}
    features = ['OFF_RATING', 'DEF_RATING', 'PACE', 'TS_PCT', 'AST_PCT', 'REB_PCT']
    
    for f in features:
        if f in df.columns:
            df[f'{f}_CURRENT_EWMA'] = df.groupby('TEAM_ID')[f].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    
    latest_games = df.groupby('TEAM_ABBREVIATION').last().reset_index()
    today = pd.to_datetime(datetime.now().date())
    
    for _, row in latest_games.iterrows():
        team = row['TEAM_ABBREVIATION']
        last_played = row['GAME_DATE']
        
        days_rest = (today - last_played).days
        days_rest = min(max(days_rest, 1), 10) 
        
        current_stats[team] = {
            'OFF_RATING_EWMA': row['OFF_RATING_CURRENT_EWMA'],
            'DEF_RATING_EWMA': row['DEF_RATING_CURRENT_EWMA'],
            'PACE_EWMA': row['PACE_CURRENT_EWMA'],
            'TS_PCT_EWMA': row['TS_PCT_CURRENT_EWMA'],
            'AST_PCT_EWMA': row['AST_PCT_CURRENT_EWMA'],
            'REB_PCT_EWMA': row['REB_PCT_CURRENT_EWMA'],
            'DAYS_REST': days_rest
        }
        
    return current_stats

def fetch_vegas_lines():
    print("Fetching live Vegas lines...")
    url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={API_KEY}&regions=us&markets=h2h,spreads,totals&bookmakers=draftkings'
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to get odds: {response.text}")
        return []
        
    games = response.json()
    todays_games = []
    
    for game in games:
        game_time = datetime.strptime(game['commence_time'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        if game_time.date() != datetime.now(timezone.utc).date():
            continue
            
        home_team = TEAM_MAP.get(game['home_team'])
        away_team = TEAM_MAP.get(game['away_team'])
        
        if not home_team or not away_team:
            continue
            
        bookie_markets = game['bookmakers'][0]['markets'] if game['bookmakers'] else []
        spread = 0
        total = 0
        
        for market in bookie_markets:
            if market['key'] == 'spreads':
                for outcome in market['outcomes']:
                    if TEAM_MAP.get(outcome['name']) == home_team:
                        spread = outcome['point'] 
            elif market['key'] == 'totals':
                total = market['outcomes'][0]['point'] 
                
        todays_games.append({
            'home': home_team,
            'away': away_team,
            'vegas_home_spread': spread,
            'vegas_total': total
        })
        
    return todays_games

def run_predictions():
    if not os.path.exists(RAW_DATA_FILE):
        print("Advanced CSV not found!")
        return
        
    df = pd.read_csv(RAW_DATA_FILE)
    home_model = joblib.load('Fun/Sports Model/xgb_home_regressor_v6.pkl')
    away_model = joblib.load('Fun/Sports Model/xgb_away_regressor_v6.pkl')
    
    current_stats = get_current_advanced_stats(df)
    games = fetch_vegas_lines()
    
    if not games:
        print("No games found for today or API key is missing.")
        return
        
    print("\n" + "="*55)
    print(" 🚀 V6 ADVANCED ALGORITHM VS VEGAS DASHBOARD 🚀 ")
    print("="*55)
    
    for game in games:
        home = game['home']
        away = game['away']
        h_stats = current_stats[home]
        a_stats = current_stats[away]
        
        # Build the exact row the XGBoost model expects (using V6 features)
        input_data = pd.DataFrame([{
            'HOME_OFF_RATING_EWMA': h_stats['OFF_RATING_EWMA'],
            'HOME_DEF_RATING_EWMA': h_stats['DEF_RATING_EWMA'],
            'HOME_PACE_EWMA': h_stats['PACE_EWMA'],
            'HOME_TS_PCT_EWMA': h_stats['TS_PCT_EWMA'],
            'HOME_AST_PCT_EWMA': h_stats['AST_PCT_EWMA'],
            'HOME_REB_PCT_EWMA': h_stats['REB_PCT_EWMA'],
            'HOME_DAYS_REST': h_stats['DAYS_REST'],
            
            'AWAY_OFF_RATING_EWMA': a_stats['OFF_RATING_EWMA'],
            'AWAY_DEF_RATING_EWMA': a_stats['DEF_RATING_EWMA'],
            'AWAY_PACE_EWMA': a_stats['PACE_EWMA'],
            'AWAY_TS_PCT_EWMA': a_stats['TS_PCT_EWMA'],
            'AWAY_AST_PCT_EWMA': a_stats['AST_PCT_EWMA'],
            'AWAY_REB_PCT_EWMA': a_stats['REB_PCT_EWMA'],
            'AWAY_DAYS_REST': a_stats['DAYS_REST']
        }])
        
        # Make sure the columns match the exact order the model was trained on
        model_cols = home_model.feature_names_in_
        input_data = input_data[model_cols]
        
        proj_home_score = home_model.predict(input_data)[0]
        proj_away_score = away_model.predict(input_data)[0]
        
        model_total = proj_home_score + proj_away_score
        model_home_spread = proj_away_score - proj_home_score 
        
        spread_edge = abs(model_home_spread - game['vegas_home_spread'])
        total_edge = abs(model_total - game['vegas_total'])
        
        favorite = home if proj_home_score > proj_away_score else away
        
        print(f"\n{away} @ {home}")
        print("-" * 35)
        print(f"Proj Score:   {home} {proj_home_score:.1f} | {away} {proj_away_score:.1f}")
        print(f"Spread Match: Model {home} {model_home_spread:+.1f} | Vegas {home} {game['vegas_home_spread']:+.1f}  ---> Diff: {spread_edge:.1f} pts")
        print(f"Total Match:  Model {model_total:.1f}  | Vegas {game['vegas_total']:.1f}  ---> Diff: {total_edge:.1f} pts")
        
        if spread_edge > 5:
            print(f"⚠️ HIGH VALUE ALERT: Spread Diff >5. Check injury reports!")
        if total_edge > 7:
            print(f"⚠️ HIGH VALUE ALERT: Total Diff >7. Check pace/injuries!")

if __name__ == "__main__":
    run_predictions()