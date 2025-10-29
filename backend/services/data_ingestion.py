"""
Data ingestion module for Clutch sports analytics.

This module handles fetching NBA data from the nba_api package,
including player game logs and team game logs. It provides
rate-limited, robust data fetching with proper error handling.
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog, TeamGameLog
from nba_api.stats.static import players, teams
from nba_api.live.nba.endpoints import ScoreBoard


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


class NBADataIngestion:
    """
    Handles NBA data ingestion with rate limiting and error handling.
    
    This class provides methods to fetch player and team game logs
    from the NBA API, with built-in rate limiting and retry logic.
    """
    
    def __init__(self, data_dir: str = "data", rate_limit_delay: float = 0.6):
        """
        Initialize the data ingestion service.
        
        Args:
            data_dir: Base directory for data storage
            rate_limit_delay: Delay between API calls in seconds
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.logs_dir = os.path.join(data_dir, "logs")
        self.rate_limit_delay = rate_limit_delay
        
        # Ensure directories exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Cache for player/team lookups
        self._player_cache = None
        self._team_cache = None
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.logs_dir, f"data_ingestion_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """Apply rate limiting between API calls."""
        time.sleep(self.rate_limit_delay)
    
    def _get_player_id(self, player_name: str) -> Optional[int]:
        """
        Get player ID from player name.
        
        Args:
            player_name: Full name of the player
            
        Returns:
            Player ID if found, None otherwise
        """
        if self._player_cache is None:
            self._player_cache = players.get_players()
        
        for player in self._player_cache:
            if player['full_name'].lower() == player_name.lower():
                return player['id']
        
        return None
    
    def _get_team_id(self, team_name: str) -> Optional[int]:
        """
        Get team ID from team name.
        
        Args:
            team_name: Full name or abbreviation of the team
            
        Returns:
            Team ID if found, None otherwise
        """
        if self._team_cache is None:
            self._team_cache = teams.get_teams()
        
        for team in self._team_cache:
            if (team['full_name'].lower() == team_name.lower() or 
                team['abbreviation'].lower() == team_name.lower()):
                return team['id']
        
        return None
    
    def fetch_player_game_log(self, player_name: str, season: str, 
                             max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch player game log data for a specific season.
        
        Args:
            player_name: Full name of the player
            season: Season in format "YYYY-YY" (e.g., "2023-24")
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame containing player game log data
            
        Raises:
            DataIngestionError: If player not found or API call fails
        """
        player_id = self._get_player_id(player_name)
        if player_id is None:
            raise DataIngestionError(f"Player '{player_name}' not found")
        
        self.logger.info(f"Fetching game log for {player_name} (ID: {player_id}) for season {season}")
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # Fetch player game log
                player_log = PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star='Regular Season'
                )
                
                df = player_log.get_data_frames()[0]
                
                if df.empty:
                    self.logger.warning(f"No data found for {player_name} in season {season}")
                    return df
                
                # Add metadata columns
                df['player_name'] = player_name
                df['player_id'] = player_id
                df['season'] = season
                df['fetched_at'] = datetime.now().isoformat()
                
                self.logger.info(f"Successfully fetched {len(df)} games for {player_name}")
                return df
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {player_name}: {str(e)}")
                if attempt == max_retries - 1:
                    raise DataIngestionError(f"Failed to fetch data for {player_name} after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        return pd.DataFrame()
    
    def fetch_team_game_log(self, team_name: str, season: str,
                           max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch team game log data for a specific season.
        
        Args:
            team_name: Full name or abbreviation of the team
            season: Season in format "YYYY-YY" (e.g., "2023-24")
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame containing team game log data
            
        Raises:
            DataIngestionError: If team not found or API call fails
        """
        team_id = self._get_team_id(team_name)
        if team_id is None:
            raise DataIngestionError(f"Team '{team_name}' not found")
        
        self.logger.info(f"Fetching game log for team {team_name} (ID: {team_id}) for season {season}")
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # Fetch team game log
                team_log = TeamGameLog(
                    team_id=team_id,
                    season=season,
                    season_type_all_star='Regular Season'
                )
                
                df = team_log.get_data_frames()[0]
                
                if df.empty:
                    self.logger.warning(f"No data found for team {team_name} in season {season}")
                    return df
                
                # Add metadata columns
                df['team_name'] = team_name
                df['team_id'] = team_id
                df['season'] = season
                df['fetched_at'] = datetime.now().isoformat()
                
                self.logger.info(f"Successfully fetched {len(df)} games for team {team_name}")
                return df
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for team {team_name}: {str(e)}")
                if attempt == max_retries - 1:
                    raise DataIngestionError(f"Failed to fetch data for team {team_name} after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        return pd.DataFrame()
    
    def save_dataframe(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save DataFrame to CSV file in the raw data directory.
        
        Args:
            df: DataFrame to save
            filename: Name of the file (without extension)
            
        Returns:
            Full path to the saved file
        """
        if df.empty:
            self.logger.warning(f"Cannot save empty DataFrame as {filename}")
            return ""
        
        filepath = os.path.join(self.raw_dir, f"{filename}.csv")
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Saved {len(df)} rows to {filepath}")
        return filepath
    
    def fetch_and_save_player_data(self, player_name: str, season: str) -> Dict[str, Any]:
        """
        Fetch player game log data and save to CSV.
        
        Args:
            player_name: Full name of the player
            season: Season in format "YYYY-YY"
            
        Returns:
            Dictionary with metadata about the operation
        """
        try:
            df = self.fetch_player_game_log(player_name, season)
            
            if df.empty:
                return {
                    'success': False,
                    'message': f'No data found for {player_name} in season {season}',
                    'filepath': None,
                    'rows': 0
                }
            
            # Create filename with sanitized player name
            safe_player_name = player_name.replace(' ', '_').replace('.', '').lower()
            filename = f"player_{safe_player_name}_{season.replace('-', '_')}"
            
            filepath = self.save_dataframe(df, filename)
            
            return {
                'success': True,
                'message': f'Successfully fetched and saved data for {player_name}',
                'filepath': filepath,
                'rows': len(df),
                'player_name': player_name,
                'season': season,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {player_name}: {str(e)}")
            return {
                'success': False,
                'message': f'Error fetching data for {player_name}: {str(e)}',
                'filepath': None,
                'rows': 0
            }
    
    def fetch_and_save_team_data(self, team_name: str, season: str) -> Dict[str, Any]:
        """
        Fetch team game log data and save to CSV.
        
        Args:
            team_name: Full name or abbreviation of the team
            season: Season in format "YYYY-YY"
            
        Returns:
            Dictionary with metadata about the operation
        """
        try:
            df = self.fetch_team_game_log(team_name, season)
            
            if df.empty:
                return {
                    'success': False,
                    'message': f'No data found for team {team_name} in season {season}',
                    'filepath': None,
                    'rows': 0
                }
            
            # Create filename with sanitized team name
            safe_team_name = team_name.replace(' ', '_').replace('.', '').lower()
            filename = f"team_{safe_team_name}_{season.replace('-', '_')}"
            
            filepath = self.save_dataframe(df, filename)
            
            return {
                'success': True,
                'message': f'Successfully fetched and saved data for team {team_name}',
                'filepath': filepath,
                'rows': len(df),
                'team_name': team_name,
                'season': season,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching data for team {team_name}: {str(e)}")
            return {
                'success': False,
                'message': f'Error fetching data for team {team_name}: {str(e)}',
                'filepath': None,
                'rows': 0
            }
    
    def verify_data_quality(self, filepath: str) -> Dict[str, Any]:
        """
        Verify the quality and structure of saved CSV data.
        
        Args:
            filepath: Path to the CSV file to verify
            
        Returns:
            Dictionary with data quality metrics
        """
        try:
            df = pd.read_csv(filepath)
            
            return {
                'filepath': filepath,
                'rows': len(df),
                'columns': list(df.columns),
                'has_nulls': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'is_readable': True
            }
            
        except Exception as e:
            return {
                'filepath': filepath,
                'is_readable': False,
                'error': str(e)
            }
