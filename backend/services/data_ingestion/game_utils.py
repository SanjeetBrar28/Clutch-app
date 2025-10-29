"""
Game utilities for fetching NBA game IDs and team metadata.

This module provides functions to fetch game IDs for a specific team and season,
along with team metadata needed for play-by-play data processing.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder
from nba_api.stats.static import teams


class GameUtils:
    """
    Utility class for fetching game IDs and team metadata.
    """
    
    def __init__(self, rate_limit_delay: float = 0.6):
        """
        Initialize GameUtils.
        
        Args:
            rate_limit_delay: Delay between API calls in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.logger = logging.getLogger(__name__)
        
        # Cache team data
        self._team_cache = None
    
    def _rate_limit(self):
        """Apply rate limiting between API calls."""
        time.sleep(self.rate_limit_delay)
    
    def _get_team_cache(self) -> List[Dict]:
        """Get cached team data."""
        if self._team_cache is None:
            self._team_cache = teams.get_teams()
        return self._team_cache
    
    def get_team_id(self, team_name: str) -> Optional[int]:
        """
        Get team ID from team name.
        
        Args:
            team_name: Full name or abbreviation of the team
            
        Returns:
            Team ID if found, None otherwise
        """
        team_cache = self._get_team_cache()
        
        for team in team_cache:
            if (team['full_name'].lower() == team_name.lower() or 
                team['abbreviation'].lower() == team_name.lower()):
                return team['id']
        
        return None
    
    def get_team_info(self, team_name: str) -> Optional[Dict]:
        """
        Get complete team information.
        
        Args:
            team_name: Full name or abbreviation of the team
            
        Returns:
            Team info dict if found, None otherwise
        """
        team_cache = self._get_team_cache()
        
        for team in team_cache:
            if (team['full_name'].lower() == team_name.lower() or 
                team['abbreviation'].lower() == team_name.lower()):
                return team
        
        return None
    
    def fetch_game_ids(self, team_name: str, season: str, 
                      max_retries: int = 3) -> List[Dict]:
        """
        Fetch all game IDs for a team in a specific season.
        
        Args:
            team_name: Full name or abbreviation of the team
            season: Season in format "YYYY-YY" (e.g., "2024-25") or "YYYY" (e.g., "2024")
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of game dictionaries with GAME_ID and metadata
            
        Raises:
            ValueError: If team not found
            RuntimeError: If API call fails after retries
        """
        team_id = self.get_team_id(team_name)
        if team_id is None:
            raise ValueError(f"Team '{team_name}' not found")
        
        team_info = self.get_team_info(team_name)
        
        # Convert season format for NBA API (use full format like "2024-25")
        if '-' in season:
            nba_season = season  # Already in correct format "2024-25"
        else:
            # Convert single year to full format "2024" -> "2024-25"
            year = int(season)
            nba_season = f"{year}-{str(year+1)[2:]}"
        
        self.logger.info(f"Fetching game IDs for {team_name} (ID: {team_id}) for season {nba_season}")
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # Use LeagueGameFinder instead of TeamGameLog
                league_games = LeagueGameFinder(
                    team_id_nullable=team_id,
                    season_nullable=nba_season
                )
                
                df = league_games.get_data_frames()[0]
                
                if df.empty:
                    self.logger.warning(f"No games found for {team_name} in season {nba_season}")
                    return []
                
                # Convert to list of dictionaries with additional metadata
                games = []
                for _, row in df.iterrows():
                    game_data = {
                        'GAME_ID': row['GAME_ID'],
                        'GAME_DATE': row['GAME_DATE'],
                        'MATCHUP': row['MATCHUP'],
                        'WL': row['WL'],
                        'PTS': row['PTS'],
                        'FG_PCT': row['FG_PCT'],
                        'FT_PCT': row['FT_PCT'],
                        'FG3_PCT': row['FG3_PCT'],
                        'REB': row['REB'],
                        'AST': row['AST'],
                        'STL': row['STL'],
                        'BLK': row['BLK'],
                        'TOV': row['TOV'],
                        'PF': row['PF'],
                        'PLUS_MINUS': row['PLUS_MINUS'],
                        'TEAM_ID': team_id,
                        'TEAM_NAME': team_name,
                        'TEAM_ABBREVIATION': team_info['abbreviation'],
                        'SEASON': season,  # Keep original format for reference
                        'NBA_SEASON': nba_season  # NBA API format
                    }
                    games.append(game_data)
                
                self.logger.info(f"Successfully fetched {len(games)} games for {team_name}")
                return games
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {team_name}: {str(e)}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to fetch game IDs for {team_name} after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        return []
    
    def get_season_summary(self, team_name: str, season: str) -> Dict:
        """
        Get a summary of the team's season performance.
        
        Args:
            team_name: Full name or abbreviation of the team
            season: Season in format "YYYY-YY"
            
        Returns:
            Dictionary with season summary statistics
        """
        games = self.fetch_game_ids(team_name, season)
        
        if not games:
            return {
                'team_name': team_name,
                'season': season,
                'total_games': 0,
                'wins': 0,
                'losses': 0,
                'win_percentage': 0.0,
                'avg_points': 0.0,
                'avg_rebounds': 0.0,
                'avg_assists': 0.0
            }
        
        df = pd.DataFrame(games)
        
        wins = len(df[df['WL'] == 'W'])
        losses = len(df[df['WL'] == 'L'])
        
        return {
            'team_name': team_name,
            'season': season,
            'total_games': len(games),
            'wins': wins,
            'losses': losses,
            'win_percentage': wins / len(games) if len(games) > 0 else 0.0,
            'avg_points': df['PTS'].mean(),
            'avg_rebounds': df['REB'].mean(),
            'avg_assists': df['AST'].mean(),
            'avg_plus_minus': df['PLUS_MINUS'].mean()
        }
