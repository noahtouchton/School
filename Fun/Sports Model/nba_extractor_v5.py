import pandas as pd
import time
import os
import random
from datetime import datetime
from nba_api.stats.endpoints import teamgamelogs

SAVE_PATH = 'Fun/Sports Model/nba_advanced_master.csv'

def get_current_season_start_year():
    now = datetime.now()
    if now.month < 8:
        return now.year - 1
    return now.year

def fetch_advanced_nba_data(start_season=2002):
    end_season = get_current_season_start_year()
    print(f"Initiating V5.1 Extraction: Standard + Advanced Metrics ({start_season} to {end_season})")
    
    all_games = []
    
    for season in range(start_season, end_season + 1):
        season_str = f"{season}-{str(season+1)[-2:]}"
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"Pulling Season: {season_str} (Attempt {attempt + 1})...")
                
                # 1. Pull Standard Stats
                std_log = teamgamelogs.TeamGameLogs(
                    season_nullable=season_str,
                    timeout=30
                ).get_data_frames()[0]
                
                # Random sleep between 2 and 4 seconds to look like a human
                time.sleep(random.uniform(2, 4)) 
                
                # 2. Pull Advanced Stats (Using the bizarre NBA API parameter name)
                adv_log = teamgamelogs.TeamGameLogs(
                    season_nullable=season_str,
                    measure_type_player_game_logs_nullable='Advanced',
                    timeout=30
                ).get_data_frames()[0]
                
                time.sleep(random.uniform(2, 4))
                
                # 3. Clean up overlapping columns so we don't get duplicates when merging
                overlap = set(std_log.columns).intersection(set(adv_log.columns))
                overlap.remove('GAME_ID')
                overlap.remove('TEAM_ID')
                adv_log_clean = adv_log.drop(columns=list(overlap))
                
                # 4. The Zipper Merge: Combine them perfectly using Game ID and Team ID
                merged_season = pd.merge(std_log, adv_log_clean, on=['GAME_ID', 'TEAM_ID'])
                
                all_games.append(merged_season)
                print(f"Successfully secured and merged {len(merged_season)} records for {season_str}.")
                break # Success! Break out of the retry loop
                
            except Exception as e:
                wait_time = (attempt + 1) * 10
                print(f"NBA Server Blocked Request. Retrying in {wait_time} seconds... (Error: {type(e).__name__})")
                time.sleep(wait_time)
        else:
            print(f"\nFailed to fetch {season_str} after {max_retries} attempts.")
            print("The NBA might have hard-banned your IP for 10 minutes. Try again later.")
            return None
                
    if not all_games:
        print("Extraction failed. No data retrieved.")
        return None
        
    master_df = pd.concat(all_games, ignore_index=True)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    master_df.to_csv(SAVE_PATH, index=False)
    
    print("\n" + "="*50)
    print(" EXTRACTION COMPLETE: ADVANCED DATA OBTAINED ")
    print(f" Total Games Processed: {len(master_df) // 2}")
    print(f" Master file saved to: {SAVE_PATH}")
    print("="*50 + "\n")
    
    return master_df

if __name__ == "__main__":
    fetch_advanced_nba_data(start_season=2002)