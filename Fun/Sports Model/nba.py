import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

RAW_DATA_FILE = 'Fun/Sports Model/nba_raw_data.csv'

def load_data():
    if not os.path.exists(RAW_DATA_FILE):
        print("Missing raw data! Run the V2 script first to pull the CSV.")
        return None
    return pd.read_csv(RAW_DATA_FILE)

def build_matchup_dataset(df):
    """
    V4 Upgrade: Includes Fatigue Tracking (Days Rest) and tracks the 
    exact Game Date so we can apply Time Decay weighting during training.
    """
    print("Crunching advanced matchup features & fatigue tracking...")
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])
    
    # Calculate Days Rest (Fatigue metric)
    df['DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
    # If it's the first game of the season, assume fully rested (7+ days). Cap at 10 to avoid offseason skew.
    df['DAYS_REST'] = df['DAYS_REST'].fillna(7).clip(upper=10) 
    
    # 1. Calculate the moving averages (The Momentum)
    features = ['PTS', 'REB', 'AST', 'FG_PCT', 'TOV', 'PLUS_MINUS']
    for f in features:
        # Tighter span=5 to react faster to recent hot/cold streaks THIS season
        df[f'{f}_EWMA'] = df.groupby('TEAM_ID')[f].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
        
    df = df.dropna()
    df['IS_HOME'] = df['MATCHUP'].str.contains(' vs. ').astype(int)
    
    # 2. Split the dataset
    home_df = df[df['IS_HOME'] == 1].copy()
    away_df = df[df['IS_HOME'] == 0].copy()
    
    home_cols = {col: f'HOME_{col}' for col in df.columns if col not in ['GAME_ID', 'GAME_DATE']}
    away_cols = {col: f'AWAY_{col}' for col in df.columns if col not in ['GAME_ID', 'GAME_DATE']}
    
    home_df = home_df.rename(columns=home_cols)
    away_df = away_df.rename(columns=away_cols)
    
    # 3. Smash them together. Keep GAME_DATE for the weighting function.
    matchups = pd.merge(home_df, away_df, on=['GAME_ID', 'GAME_DATE'])
    
    return matchups

def train_dual_regressors(df):
    """
    V4 Upgrade: Implements Exponential Time Decay weighting. 
    Forces the model to prioritize modern trends while retaining historical baseline knowledge.
    """
    print("Training Time-Weighted XGBoost Regressors...")
    
    feature_cols = [
        'HOME_PTS_EWMA', 'HOME_REB_EWMA', 'HOME_AST_EWMA', 'HOME_FG_PCT_EWMA', 'HOME_TOV_EWMA', 'HOME_PLUS_MINUS_EWMA', 'HOME_DAYS_REST',
        'AWAY_PTS_EWMA', 'AWAY_REB_EWMA', 'AWAY_AST_EWMA', 'AWAY_FG_PCT_EWMA', 'AWAY_TOV_EWMA', 'AWAY_PLUS_MINUS_EWMA', 'AWAY_DAYS_REST'
    ]
    
    X = df[feature_cols]
    y_home = df['HOME_PTS']
    y_away = df['AWAY_PTS']
    
    # --- THE MAGIC SAUCE: EXPONENTIAL TIME DECAY WEIGHTING ---
    max_date = df['GAME_DATE'].max()
    days_old = (max_date - df['GAME_DATE']).dt.days
    
    # Math translation: Games from 10 years ago get a weight near 0.1. 
    # Games from this season get heavily boosted.
    sample_weights = np.exp(-days_old / 1500) 
    
    # Supercharge the current season (games played in the last 180 days get a 3x multiplier)
    current_season_mask = days_old < 180
    sample_weights = np.where(current_season_mask, sample_weights * 3, sample_weights)
    # ---------------------------------------------------------
    
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test, w_train, w_test = train_test_split(
        X, y_home, y_away, sample_weights, test_size=0.15, random_state=42, shuffle=False
    )
    
    # Increased complexity slightly since you have the CPU power
    home_model = xgb.XGBRegressor(n_estimators=700, learning_rate=0.01, max_depth=7, random_state=42, subsample=0.8)
    away_model = xgb.XGBRegressor(n_estimators=700, learning_rate=0.01, max_depth=7, random_state=42, subsample=0.8)
    
    # Fit the models USING the sample weights
    home_model.fit(X_train, y_home_train, sample_weight=w_train)
    away_model.fit(X_train, y_away_train, sample_weight=w_train)
    
    home_preds = home_model.predict(X_test)
    away_preds = away_model.predict(X_test)
    
    print(f"Home Score Mean Absolute Error: {mean_absolute_error(y_home_test, home_preds):.2f} points")
    print(f"Away Score Mean Absolute Error: {mean_absolute_error(y_away_test, away_preds):.2f} points")
    
    joblib.dump(home_model, 'Fun/Sports Model/xgb_home_regressor_v4.pkl')
    joblib.dump(away_model, 'Fun/Sports Model/xgb_away_regressor_v4.pkl')
    
    return home_model, away_model, df

def predict_game(home_abbr, away_abbr, historical_df, home_model, away_model):
    """
    Takes two team abbreviations, grabs their most recent momentum stats, 
    and projects the Vegas lines. Updated to include Fatigue Tracking.
    """
    # Get the latest stats for the requested teams
    home_stats = historical_df[historical_df['HOME_TEAM_ABBREVIATION'] == home_abbr].iloc[-1]
    away_stats = historical_df[historical_df['AWAY_TEAM_ABBREVIATION'] == away_abbr].iloc[-1]
    
    # Build the input row exactly as the model was trained on it
    input_data = pd.DataFrame([{
        'HOME_PTS_EWMA': home_stats['HOME_PTS_EWMA'],
        'HOME_REB_EWMA': home_stats['HOME_REB_EWMA'],
        'HOME_AST_EWMA': home_stats['HOME_AST_EWMA'],
        'HOME_FG_PCT_EWMA': home_stats['HOME_FG_PCT_EWMA'],
        'HOME_TOV_EWMA': home_stats['HOME_TOV_EWMA'],
        'HOME_PLUS_MINUS_EWMA': home_stats['HOME_PLUS_MINUS_EWMA'],
        'HOME_DAYS_REST': home_stats['HOME_DAYS_REST'],  # <--- NEW V4 FEATURE
        
        'AWAY_PTS_EWMA': away_stats['AWAY_PTS_EWMA'],
        'AWAY_REB_EWMA': away_stats['AWAY_REB_EWMA'],
        'AWAY_AST_EWMA': away_stats['AWAY_AST_EWMA'],
        'AWAY_FG_PCT_EWMA': away_stats['AWAY_FG_PCT_EWMA'],
        'AWAY_TOV_EWMA': away_stats['AWAY_TOV_EWMA'],
        'AWAY_PLUS_MINUS_EWMA': away_stats['AWAY_PLUS_MINUS_EWMA'],
        'AWAY_DAYS_REST': away_stats['AWAY_DAYS_REST']   # <--- NEW V4 FEATURE
    }])
    
    proj_home_score = home_model.predict(input_data)[0]
    proj_away_score = away_model.predict(input_data)[0]
    
    proj_total = proj_home_score + proj_away_score
    proj_spread = abs(proj_home_score - proj_away_score)
    favorite = home_abbr if proj_home_score > proj_away_score else away_abbr
    
    print("\n" + "="*40)
    print(f" THE ALGORITHM SPEAKS: {away_abbr} @ {home_abbr}")
    print("="*40)
    print(f"Projected Score: {home_abbr} {proj_home_score:.1f} | {away_abbr} {proj_away_score:.1f}")
    print(f"Moneyline Pick:  {favorite} Wins")
    print(f"Projected Spread:{favorite} -{proj_spread:.1f}")
    print(f"Projected Total: {proj_total:.1f} (Over/Under)")
    print("="*40 + "\n")

if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        matchup_df = build_matchup_dataset(raw_df)
        home_mod, away_mod, processed_df = train_dual_regressors(matchup_df)
        
        # Test it out! Use standard 3-letter NBA abbreviations (e.g., LAL, BOS, MIA, DEN)
        #predict_game("LAK", "CHA", processed_df, home_mod, away_mod)
        predict_game("CLE", "ORL", processed_df, home_mod, away_mod)