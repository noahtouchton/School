import pandas as pd
import time
import os
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog
from requests.exceptions import ReadTimeout, ConnectionError

# Save exactly where your Mac expects it
SAVE_PATH = 'Fun/Sports Model/nba_raw_data.csv'

def get_current_season_start_year():
    """Calculates the current NBA season based on today's date."""
    now = datetime.now()
    # If we are in Jan-July, the season started the previous year
    if now.month < 8:
        return now.year - 1
    return now.year

def fetch_all_nba_data(start_season=2002):
    """
    Pulls every game log from start_season up to today.
    Includes auto-retry logic to defeat NBA API rate limits.
    """
    end_season = get_current_season_start_year()
    print(f"Initiating Master Extraction Protocol: {start_season} to {end_season}...")
    
    all_games = []
    
    for season in range(start_season, end_season + 1):
        # Format for the API (e.g., 2025 -> "2025-26")
        season_str = f"{season}-{str(season+1)[-2:]}"
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"Pulling Season: {season_str} (Attempt {attempt + 1})...")
                # Pull the data for all teams ('T')
                game_log = leaguegamelog.LeagueGameLog(
                    season=season_str, 
                    player_or_team_abbreviation='T',
                    timeout=30 # 30 second timeout so it doesn't hang forever
                ).get_data_frames()[0]
                
                all_games.append(game_log)
                print(f"Successfully secured {len(game_log)} game records for {season_str}.")
                
                # Sleep to respect the API rate limit before the next season
                time.sleep(2)
                break # Break out of the retry loop if successful
                
            except (ReadTimeout, ConnectionError) as e:
                wait_time = (attempt + 1) * 5
                print(f"API Connection lost. Retrying in {wait_time} seconds... ({e})")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Critical error on {season_str}: {e}")
                break # Move to the next season if it's a non-network error
                
    if not all_games:
        print("Extraction failed. No data retrieved.")
        return None
        
    # Combine all seasons into one massive DataFrame
    master_df = pd.concat(all_games, ignore_index=True)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # Save to CSV
    master_df.to_csv(SAVE_PATH, index=False)
    print("\n" + "="*50)
    print(f"EXTRACTION COMPLETE!")
    print(f"Total Games Processed: {len(master_df) // 2} (Two rows per game)")
    print(f"Master file saved to: {SAVE_PATH}")
    print("="*50 + "\n")
    
    return master_df

if __name__ == "__main__":
    fetch_all_nba_data(start_season=2002)