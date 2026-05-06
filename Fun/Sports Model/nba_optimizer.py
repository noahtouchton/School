import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

RAW_DATA_FILE = 'Fun/Sports Model/nba_raw_data.csv'

# Disable Optuna's excessive logging so we only see the important stuff
optuna.logging.set_verbosity(optuna.logging.WARNING)

def build_matchup_dataset(df):
    """Rebuilding the exact same dataset so the optimization matches reality."""
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])
    
    df['DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
    df['DAYS_REST'] = df['DAYS_REST'].fillna(7).clip(upper=10) 
    
    features = ['PTS', 'REB', 'AST', 'FG_PCT', 'TOV', 'PLUS_MINUS']
    for f in features:
        df[f'{f}_EWMA'] = df.groupby('TEAM_ID')[f].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
        
    df = df.dropna()
    df['IS_HOME'] = df['MATCHUP'].str.contains(' vs. ').astype(int)
    
    home_df = df[df['IS_HOME'] == 1].copy()
    away_df = df[df['IS_HOME'] == 0].copy()
    
    home_cols = {col: f'HOME_{col}' for col in df.columns if col not in ['GAME_ID', 'GAME_DATE']}
    away_cols = {col: f'AWAY_{col}' for col in df.columns if col not in ['GAME_ID', 'GAME_DATE']}
    
    home_df = home_df.rename(columns=home_cols)
    away_df = away_df.rename(columns=away_cols)
    
    matchups = pd.merge(home_df, away_df, on=['GAME_ID', 'GAME_DATE'])
    return matchups

def prepare_data():
    df = pd.read_csv(RAW_DATA_FILE)
    df = build_matchup_dataset(df)
    
    feature_cols = [
        'HOME_PTS_EWMA', 'HOME_REB_EWMA', 'HOME_AST_EWMA', 'HOME_FG_PCT_EWMA', 'HOME_TOV_EWMA', 'HOME_PLUS_MINUS_EWMA', 'HOME_DAYS_REST',
        'AWAY_PTS_EWMA', 'AWAY_REB_EWMA', 'AWAY_AST_EWMA', 'AWAY_FG_PCT_EWMA', 'AWAY_TOV_EWMA', 'AWAY_PLUS_MINUS_EWMA', 'AWAY_DAYS_REST'
    ]
    
    X = df[feature_cols]
    y_home = df['HOME_PTS']
    
    # Time Decay Weights
    max_date = df['GAME_DATE'].max()
    days_old = (max_date - df['GAME_DATE']).dt.days
    sample_weights = np.exp(-days_old / 1500) 
    current_season_mask = days_old < 180
    sample_weights = np.where(current_season_mask, sample_weights * 3, sample_weights)
    
    # Split the data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_home, sample_weights, test_size=0.15, random_state=42, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, w_train

# Global variables so the objective function can see them
X_train, X_test, y_train, y_test, w_train = prepare_data()

def objective(trial):
    """
    The mathematical hunting ground. Optuna will test values within these ranges.
    """
    param = {
        # 'hist' is insanely fast on multi-core Apple Silicon
        'tree_method': 'hist', 
        'random_state': 42,
        
        # The parameters we are forcing the M5 Max to optimize
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0)
    }
    
    model = xgb.XGBRegressor(**param)
    
    # Fit with our time-decay weights
    model.fit(X_train, y_train, sample_weight=w_train)
    
    # Predict and calculate MAE
    preds = model.predict(X_test)
    error = mean_absolute_error(y_test, preds)
    
    return error

if __name__ == "__main__":
    print(f"Firing up the M5 Max. Initializing Bayesian Optimization...")
    print("Testing 150 different neural pathways. This will take some time...")
    
    # Create the study. We want to MINIMIZE the error.
    study = optuna.create_study(direction='minimize')
    
    # n_jobs=-1 tells Optuna to use every available CPU thread your Mac has
    study.optimize(objective, n_trials=150, n_jobs=-1, show_progress_bar=True)
    
    print("\n" + "="*50)
    print(" 🎯 OPTIMIZATION COMPLETE 🎯 ")
    print("="*50)
    print(f"Best Error Achieved (MAE): {study.best_value:.3f} points")
    print("The Mathematically Optimal Parameters:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")