"""
Play-by-play data fetcher using the playbyplayv3 endpoint.

This module handles fetching raw play-by-play data for individual games
and saving it to CSV files with proper error handling and rate limiting.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
from nba_api.stats.endpoints import PlayByPlayV3


class PlayByPlayFetcher:
    """
    Handles fetching play-by-play data for individual games.
    """
    
    def __init__(self, output_dir: str = "data/raw/playbyplay", 
                 rate_limit_delay: float = 0.6):
        """
        Initialize PlayByPlayFetcher.
        
        Args:
            output_dir: Base directory to save raw play-by-play files
            rate_limit_delay: Delay between API calls in seconds
        """
        self.output_dir = output_dir
        self.rate_limit_delay = rate_limit_delay
        self.logger = logging.getLogger(__name__)
        
        # Ensure base output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _rate_limit(self):
        """Apply rate limiting between API calls."""
        time.sleep(self.rate_limit_delay)
    
    def fetch_game_playbyplay(self, game_id: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch play-by-play data for a specific game.
        
        Args:
            game_id: NBA game ID
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame with play-by-play data, or None if failed
        """
        self.logger.info(f"Fetching play-by-play data for game {game_id}")
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # Fetch play-by-play data
                pbp = PlayByPlayV3(game_id=game_id)
                
                # Get the play-by-play DataFrame
                df = pbp.get_data_frames()[0]
                
                if df.empty:
                    self.logger.warning(f"No play-by-play data found for game {game_id}")
                    return None
                
                # Add metadata columns
                df['GAME_ID'] = game_id
                df['fetched_at'] = pd.Timestamp.now().isoformat()
                
                # Map the actual column names to our expected names
                column_mapping = {
                    'gameId': 'GAME_ID',
                    'actionNumber': 'EVENTNUM', 
                    'period': 'PERIOD',
                    'clock': 'PCTIMESTRING',
                    'scoreHome': 'HOME_SCORE',
                    'scoreAway': 'AWAY_SCORE',
                    'description': 'HOMEDESCRIPTION',  # This might need adjustment
                    'playerName': 'PLAYER1_NAME',
                    'teamTricode': 'PLAYER1_TEAM_ABBREVIATION',
                    'actionType': 'EVENTMSGTYPE',
                    'subType': 'EVENTMSGACTIONTYPE'
                }
                
                # Rename columns if they exist
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns:
                        df[new_name] = df[old_name]
                
                self.logger.info(f"Successfully fetched {len(df)} plays for game {game_id}")
                return df
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for game {game_id}: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to fetch play-by-play data for game {game_id} after {max_retries} attempts: {str(e)}")
                    return None
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        return None
    
    def save_game_data(self, df: pd.DataFrame, game_id: str, team_name: str, season: str) -> str:
        """
        Save play-by-play DataFrame to CSV file with organized directory structure.
        
        Args:
            df: DataFrame with play-by-play data
            game_id: NBA game ID
            team_name: Team name for directory organization
            season: Season for directory organization
            
        Returns:
            Path to the saved file
        """
        if df is None or df.empty:
            self.logger.warning(f"Cannot save empty DataFrame for game {game_id}")
            return ""
        
        # Create organized directory structure: team_name/season/
        safe_team_name = team_name.replace(' ', '_').replace('.', '').lower()
        safe_season = season.replace('-', '_')
        
        team_season_dir = os.path.join(self.output_dir, safe_team_name, safe_season)
        os.makedirs(team_season_dir, exist_ok=True)
        
        filename = f"{game_id}.csv"
        filepath = os.path.join(team_season_dir, filename)
        
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Saved {len(df)} plays to {filepath}")
        return filepath
    
    def fetch_and_save_game(self, game_id: str, team_name: str, season: str) -> Dict:
        """
        Fetch play-by-play data for a game and save to CSV.
        
        Args:
            game_id: NBA game ID
            team_name: Team name for directory organization
            season: Season for directory organization
            
        Returns:
            Dictionary with operation results
        """
        try:
            df = self.fetch_game_playbyplay(game_id)
            
            if df is None or df.empty:
                return {
                    'success': False,
                    'game_id': game_id,
                    'message': f'No data found for game {game_id}',
                    'filepath': None,
                    'plays': 0
                }
            
            filepath = self.save_game_data(df, game_id, team_name, season)
            
            return {
                'success': True,
                'game_id': game_id,
                'message': f'Successfully fetched and saved data for game {game_id}',
                'filepath': filepath,
                'plays': len(df),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching data for game {game_id}: {str(e)}")
            return {
                'success': False,
                'game_id': game_id,
                'message': f'Error fetching data for game {game_id}: {str(e)}',
                'filepath': None,
                'plays': 0
            }
    
    def fetch_multiple_games(self, game_ids: List[str], team_name: str, season: str) -> List[Dict]:
        """
        Fetch play-by-play data for multiple games.
        
        Args:
            game_ids: List of NBA game IDs
            team_name: Team name for directory organization
            season: Season for directory organization
            
        Returns:
            List of operation results for each game
        """
        results = []
        
        self.logger.info(f"Starting to fetch play-by-play data for {len(game_ids)} games")
        
        for i, game_id in enumerate(game_ids, 1):
            self.logger.info(f"Processing game {i}/{len(game_ids)}: {game_id}")
            
            result = self.fetch_and_save_game(game_id, team_name, season)
            results.append(result)
            
            # Log progress
            if result['success']:
                self.logger.info(f"✅ Game {game_id}: {result['plays']} plays saved")
            else:
                self.logger.error(f"❌ Game {game_id}: {result['message']}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        total_plays = sum(r['plays'] for r in results)
        
        self.logger.info(f"Completed fetching: {successful}/{len(game_ids)} games successful, {total_plays} total plays")
        
        return results
    
    def get_fetch_summary(self, results: List[Dict]) -> Dict:
        """
        Get a summary of fetch results.
        
        Args:
            results: List of fetch results
            
        Returns:
            Dictionary with summary statistics
        """
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        total_plays = sum(r['plays'] for r in successful)
        
        return {
            'total_games': len(results),
            'successful_games': len(successful),
            'failed_games': len(failed),
            'success_rate': len(successful) / len(results) if results else 0,
            'total_plays': total_plays,
            'avg_plays_per_game': total_plays / len(successful) if successful else 0,
            'failed_game_ids': [r['game_id'] for r in failed]
        }
