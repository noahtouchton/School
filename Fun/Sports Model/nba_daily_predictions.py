import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import datetime, timezone

# --- CONFIGURATION ---
# Paste your API key from The Odds API here
API_KEY = '2c53c9affcb2d9c702b0538ba851f206' 
RAW_DATA_FILE = 'Fun/Sports Model/nba_raw_data.csv'

# Dictionary to translate Vegas full names to NBA abbreviations
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

def get_current_team_stats(df):
    """
    Calculates the CURRENT momentum (EWMA) and Days Rest for all 30 teams 
    leading into tonight's games.
    """
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])
    
    current_stats = {}
    features = ['PTS', 'REB', 'AST', 'FG_PCT', 'TOV', 'PLUS_MINUS']
    
    # Calculate current momentum (notice we don't use .shift(1) here because 
    # we want the EWMA *including* their most recently played game to predict the NEXT one)
    for f in features:
        df[f'{f}_CURRENT_EWMA'] = df.groupby('TEAM_ID')[f].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    
    # Extract the absolute latest row for each team
    latest_games = df.groupby('TEAM_ABBREVIATION').last().reset_index()
    
    today = pd.to_datetime(datetime.now().date())
    
    for _, row in latest_games.iterrows():
        team = row['TEAM_ABBREVIATION']
        last_played = row['GAME_DATE']
        
        # Calculate Fatigue (Days Rest)
        days_rest = (today - last_played).days
        days_rest = min(max(days_rest, 1), 10) # Cap between 1 and 10 days
        
        current_stats[team] = {
            'PTS_EWMA': row['PTS_CURRENT_EWMA'],
            'REB_EWMA': row['REB_CURRENT_EWMA'],
            'AST_EWMA': row['AST_CURRENT_EWMA'],
            'FG_PCT_EWMA': row['FG_PCT_CURRENT_EWMA'],
            'TOV_EWMA': row['TOV_CURRENT_EWMA'],
            'PLUS_MINUS_EWMA': row['PLUS_MINUS_CURRENT_EWMA'],
            'DAYS_REST': days_rest
        }
        
    return current_stats

def fetch_vegas_lines():
    """Pulls tonight's NBA games and odds from The Odds API."""
    print("Fetching live Vegas lines...")
    url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={API_KEY}&regions=us&markets=h2h,spreads,totals&bookmakers=draftkings'
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to get odds: {response.text}")
        return []
        
    games = response.json()
    todays_games = []
    
    for game in games:
        # Only grab games happening today/tonight
        game_time = datetime.strptime(game['commence_time'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        if game_time.date() != datetime.now(timezone.utc).date():
            continue
            
        home_team = TEAM_MAP.get(game['home_team'])
        away_team = TEAM_MAP.get(game['away_team'])
        
        if not home_team or not away_team:
            continue
            
        # Parse the DraftKings odds
        bookie = game['bookmakers'][0]['markets']
        spread = 0
        total = 0
        
        for market in bookie:
            if market['key'] == 'spreads':
                for outcome in market['outcomes']:
                    if TEAM_MAP.get(outcome['name']) == home_team:
                        spread = outcome['point'] # Home team's spread
            elif market['key'] == 'totals':
                total = market['outcomes'][0]['point'] # Over/Under line
                
        todays_games.append({
            'home': home_team,
            'away': away_team,
            'vegas_home_spread': spread,
            'vegas_total': total
        })
        
    return todays_games

def run_predictions():
    # 1. Load Data and Models
    if not os.path.exists(RAW_DATA_FILE):
        print("Master CSV not found!")
        return
        
    df = pd.read_csv(RAW_DATA_FILE)
    home_model = joblib.load('Fun/Sports Model/xgb_home_regressor_v4.pkl')
    away_model = joblib.load('Fun/Sports Model/xgb_away_regressor_v4.pkl')
    
    # 2. Get Current Stats
    current_stats = get_current_team_stats(df)
    
    # 3. Get Tonight's Games
    games = fetch_vegas_lines()
    if not games:
        print("No games found for today or API key is missing/invalid.")
        return
        
    print("\n" + "="*50)
    print(" 🏀 TONIGHT'S ALGORITHM VS VEGAS DASHBOARD 🏀 ")
    print("="*50)
    
    for game in games:
        home = game['home']
        away = game['away']
        h_stats = current_stats[home]
        a_stats = current_stats[away]
        
        # Build the exact row the XGBoost model expects
        input_data = pd.DataFrame([{
            'HOME_PTS_EWMA': h_stats['PTS_EWMA'],
            'HOME_REB_EWMA': h_stats['REB_EWMA'],
            'HOME_AST_EWMA': h_stats['AST_EWMA'],
            'HOME_FG_PCT_EWMA': h_stats['FG_PCT_EWMA'],
            'HOME_TOV_EWMA': h_stats['TOV_EWMA'],
            'HOME_PLUS_MINUS_EWMA': h_stats['PLUS_MINUS_EWMA'],
            'HOME_DAYS_REST': h_stats['DAYS_REST'],
            
            'AWAY_PTS_EWMA': a_stats['PTS_EWMA'],
            'AWAY_REB_EWMA': a_stats['REB_EWMA'],
            'AWAY_AST_EWMA': a_stats['AST_EWMA'],
            'AWAY_FG_PCT_EWMA': a_stats['FG_PCT_EWMA'],
            'AWAY_TOV_EWMA': a_stats['TOV_EWMA'],
            'AWAY_PLUS_MINUS_EWMA': a_stats['PLUS_MINUS_EWMA'],
            'AWAY_DAYS_REST': a_stats['DAYS_REST']
        }])
        
        proj_home_score = home_model.predict(input_data)[0]
        proj_away_score = away_model.predict(input_data)[0]
        
        # Calculate Model Lines
        model_total = proj_home_score + proj_away_score
        model_home_spread = proj_away_score - proj_home_score # If home scores 110 and away scores 100, spread is -10
        
        # Calculate Edges
        spread_edge = abs(model_home_spread - game['vegas_home_spread'])
        total_edge = abs(model_total - game['vegas_total'])
        
        favorite = home if proj_home_score > proj_away_score else away
        
        print(f"\n{away} @ {home}")
        print("-" * 30)
        print(f"Proj Score:   {home} {proj_home_score:.1f} | {away} {proj_away_score:.1f}")
        print(f"Spread Match: Model {home} {model_home_spread:+.1f} | Vegas {home} {game['vegas_home_spread']:+.1f}  ---> Diff: {spread_edge:.1f} pts")
        print(f"Total Match:  Model {model_total:.1f}  | Vegas {game['vegas_total']:.1f}  ---> Diff: {total_edge:.1f} pts")
        
        # Flag massive discrepancies for review
        if spread_edge > 5:
            print(f"⚠️ HIGH VALUE ALERT: Vegas disagrees on Spread by >5 points. Check injury reports!")
        if total_edge > 7:
            print(f"⚠️ HIGH VALUE ALERT: Vegas disagrees on Total by >7 points. Check pace/injuries!")

if __name__ == "__main__":
    run_predictions()