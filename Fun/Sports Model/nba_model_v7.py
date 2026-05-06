import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

RAW_DATA_FILE = 'Fun/Sports Model/nba_advanced_master.csv'

def build_v7_matchup_dataset(df):
    print("Engineering V7: Margin Predictor + Four Factors...")
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])
    
    # Calculate Fatigue
    df['DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
    df['DAYS_REST'] = df['DAYS_REST'].fillna(7).clip(upper=10) 
    
    # Four Factors + Efficiency Metrics
    # EFG_PCT and FTA_RATE are usually in the 'Advanced' or 'Base' logs
    features = [
        'OFF_RATING', 'DEF_RATING', 'PACE', 
        'EFG_PCT', 'TM_TOV_PCT', 'OREB_PCT', 'FTA_RATE'
    ]
    
    for f in features:
        if f in df.columns:
            df[f'{f}_EWMA'] = df.groupby('TEAM_ID')[f].transform(
                lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
            )
        
    df = df.dropna(subset=[f'{f}_EWMA' for f in features if f in df.columns])
    df['IS_HOME'] = df['MATCHUP'].str.contains(' vs. ').astype(int)
    
    home_df = df[df['IS_HOME'] == 1].copy()
    away_df = df[df['IS_HOME'] == 0].copy()
    
    # Rename for merge
    home_cols = {col: f'HOME_{col}' for col in df.columns if col not in ['GAME_ID', 'GAME_DATE']}
    away_cols = {col: f'AWAY_{col}' for col in df.columns if col not in ['GAME_ID', 'GAME_DATE']}
    
    matchups = pd.merge(home_df.rename(columns=home_cols), 
                        away_df.rename(columns=away_cols), 
                        on=['GAME_ID', 'GAME_DATE'])
    
    # THE NEW TARGET: Home Margin
    matchups['HOME_MARGIN_ACTUAL'] = matchups['HOME_PTS'] - matchups['AWAY_PTS']
    # THE SECONDARY TARGET: Total Points (for Over/Unders)
    matchups['TOTAL_PTS_ACTUAL'] = matchups['HOME_PTS'] + matchups['AWAY_PTS']
    
    return matchups

def train_v7_models(df):
    print("Training High-Precision Margin & Total Regressors...")
    
    feature_cols = [col for col in df.columns if '_EWMA' in col or '_DAYS_REST' in col]
    
    X = df[feature_cols]
    y_margin = df['HOME_MARGIN_ACTUAL']
    y_total = df['TOTAL_PTS_ACTUAL']
    
    # Time Decay Weights
    max_date = df['GAME_DATE'].max()
    days_old = (max_date - df['GAME_DATE']).dt.days
    sample_weights = np.exp(-days_old / 1500) 
    sample_weights = np.where(days_old < 180, sample_weights * 3, sample_weights)
    
    # Split for Margin
    X_train, X_test, y_m_train, y_m_test, w_train, w_test = train_test_split(
        X, y_margin, sample_weights, test_size=0.15, random_state=42, shuffle=False
    )
    
    # Split for Total
    _, _, y_t_train, y_t_test, _, _ = train_test_split(
        X, y_total, sample_weights, test_size=0.15, random_state=42, shuffle=False
    )
    
    # Hardcoded Optimized Params from your Optuna run
    params = {
        'n_estimators': 721, 'learning_rate': 0.0076, 'max_depth': 8,
        'subsample': 0.79, 'colsample_bytree': 0.96, 'gamma': 1.8
    }
    
    margin_model = xgb.XGBRegressor(**params)
    total_model = xgb.XGBRegressor(**params)
    
    margin_model.fit(X_train, y_m_train, sample_weight=w_train)
    total_model.fit(X_train, y_t_train, sample_weight=w_train)
    
    m_preds = margin_model.predict(X_test)
    t_preds = total_model.predict(X_test)
    
    print(f"\nMargin Mean Absolute Error: {mean_absolute_error(y_m_test, m_preds):.2f} pts")
    print(f"Total Points Mean Absolute Error: {mean_absolute_error(y_t_test, t_preds):.2f} pts")
    
    joblib.dump(margin_model, 'Fun/Sports Model/nba_margin_v7.pkl')
    joblib.dump(total_model, 'Fun/Sports Model/nba_total_v7.pkl')
    
    return margin_model, total_model

if __name__ == "__main__":
    raw_df = pd.read_csv(RAW_DATA_FILE)
    matchup_df = build_v7_matchup_dataset(raw_df)
    train_v7_models(matchup_df)