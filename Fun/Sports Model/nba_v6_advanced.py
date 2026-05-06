import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Pointing to the NEW Advanced Metrics dataset
RAW_DATA_FILE = 'Fun/Sports Model/nba_advanced_master.csv'

def load_data():
    if not os.path.exists(RAW_DATA_FILE):
        print(f"Missing {RAW_DATA_FILE}! Run the V5 Advanced Extractor first.")
        return None
    return pd.read_csv(RAW_DATA_FILE)

def build_advanced_matchup_dataset(df):
    """
    V6 Upgrade: Swapping raw stats for Pace-Adjusted Efficiency Metrics.
    """
    print("Crunching Pace and Efficiency features...")
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])
    
    # Fatigue tracking
    df['DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
    df['DAYS_REST'] = df['DAYS_REST'].fillna(7).clip(upper=10) 
    
    # THE FUEL UPGRADE: We now use Ratings and Percentages instead of raw counting stats
    # Note: Column names match the NBA API 'Advanced' and 'Base' endpoints
    features = ['OFF_RATING', 'DEF_RATING', 'PACE', 'TS_PCT', 'AST_PCT', 'REB_PCT']
    
    for f in features:
        # Tighter span=5 to react fast to recent streaks
        if f in df.columns:
            df[f'{f}_EWMA'] = df.groupby('TEAM_ID')[f].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
        else:
            print(f"Warning: {f} not found in dataset. Check extraction.")
        
    df = df.dropna(subset=[f'{f}_EWMA' for f in features if f in df.columns])
    df['IS_HOME'] = df['MATCHUP'].str.contains(' vs. ').astype(int)
    
    home_df = df[df['IS_HOME'] == 1].copy()
    away_df = df[df['IS_HOME'] == 0].copy()
    
    home_cols = {col: f'HOME_{col}' for col in df.columns if col not in ['GAME_ID', 'GAME_DATE']}
    away_cols = {col: f'AWAY_{col}' for col in df.columns if col not in ['GAME_ID', 'GAME_DATE']}
    
    home_df = home_df.rename(columns=home_cols)
    away_df = away_df.rename(columns=away_cols)
    
    matchups = pd.merge(home_df, away_df, on=['GAME_ID', 'GAME_DATE'])
    return matchups

def train_optimized_regressors(df):
    """
    V6 Upgrade: Using the M5 Max Optimized Hyperparameters.
    """
    print("Training Optimized XGBoost Regressors with Advanced Metrics...")
    
    # Dynamically grab the engineered columns so we don't hardcode them
    feature_cols = [col for col in df.columns if '_EWMA' in col or '_DAYS_REST' in col]
    
    X = df[feature_cols]
    y_home = df['HOME_PTS']
    y_away = df['AWAY_PTS']
    
    # Exponential Time Decay Weighting (supercharging the last 180 days)
    max_date = df['GAME_DATE'].max()
    days_old = (max_date - df['GAME_DATE']).dt.days
    sample_weights = np.exp(-days_old / 1500) 
    current_season_mask = days_old < 180
    sample_weights = np.where(current_season_mask, sample_weights * 3, sample_weights)
    
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test, w_train, w_test = train_test_split(
        X, y_home, y_away, sample_weights, test_size=0.15, random_state=42, shuffle=False
    )
    
    # THE ENGINE UPGRADE: Your exact Optuna results
    optimized_params = {
        'tree_method': 'hist',
        'random_state': 42,
        'n_estimators': 721,
        'learning_rate': 0.007640786446311546,
        'max_depth': 8,
        'subsample': 0.7980920646761279,
        'colsample_bytree': 0.9697879741624734,
        'min_child_weight': 4,
        'gamma': 1.8079246340418287
    }
    
    home_model = xgb.XGBRegressor(**optimized_params)
    away_model = xgb.XGBRegressor(**optimized_params)
    
    home_model.fit(X_train, y_home_train, sample_weight=w_train)
    away_model.fit(X_train, y_away_train, sample_weight=w_train)
    
    home_preds = home_model.predict(X_test)
    away_preds = away_model.predict(X_test)
    
    print(f"\nFinal Home Score Mean Absolute Error: {mean_absolute_error(y_home_test, home_preds):.2f} points")
    print(f"Final Away Score Mean Absolute Error: {mean_absolute_error(y_away_test, away_preds):.2f} points")
    
    joblib.dump(home_model, 'Fun/Sports Model/xgb_home_regressor_v6.pkl')
    joblib.dump(away_model, 'Fun/Sports Model/xgb_away_regressor_v6.pkl')
    print("V6 Models saved successfully!")
    
    return home_model, away_model, df

if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        matchup_df = build_advanced_matchup_dataset(raw_df)
        train_optimized_regressors(matchup_df)