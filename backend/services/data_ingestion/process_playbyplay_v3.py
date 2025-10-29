"""
Play-by-play data processor for cleaning and merging game data.

This module handles loading raw play-by-play CSV files, cleaning and standardizing
columns, and merging all games into a single processed dataset.
"""

import os
import logging
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class PlayByPlayProcessor:
    """
    Handles processing and cleaning play-by-play data.
    """
    
    def __init__(self, raw_dir: str = "data/raw/playbyplay", 
                 processed_dir: str = "data/processed"):
        """
        Initialize PlayByPlayProcessor.
        
        Args:
            raw_dir: Directory containing raw play-by-play CSV files
            processed_dir: Directory to save processed data
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.logger = logging.getLogger(__name__)
        
        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Core columns to retain
        self.core_columns = [
            'GAME_ID', 'EVENTNUM', 'PERIOD', 'PCTIMESTRING', 'SCOREMARGIN',
            'HOMEDESCRIPTION', 'VISITORDESCRIPTION',
            'PLAYER1_NAME', 'PLAYER1_TEAM_ABBREVIATION', 'PLAYER2_NAME',
            'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE', 'PERSON1TYPE', 
            'TEAM_ID_HOME', 'TEAM_ID_AWAY'
        ]
    
    def load_game_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load play-by-play data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with game data, or None if failed
        """
        try:
            df = pd.read_csv(filepath)
            
            if df.empty:
                self.logger.warning(f"Empty file: {filepath}")
                return None
            
            self.logger.info(f"Loaded {len(df)} plays from {os.path.basename(filepath)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {str(e)}")
            return None
    
    def load_all_games(self, game_ids: List[str], team_name: str, season: str) -> List[pd.DataFrame]:
        """
        Load play-by-play data for all games from organized directory structure.
        
        Args:
            game_ids: List of game IDs to load
            team_name: Team name for directory organization
            season: Season for directory organization
            
        Returns:
            List of DataFrames with game data
        """
        games_data = []
        
        # Create organized directory path
        safe_team_name = team_name.replace(' ', '_').replace('.', '').lower()
        safe_season = season.replace('-', '_')
        team_season_dir = os.path.join(self.raw_dir, safe_team_name, safe_season)
        
        self.logger.info(f"Loading play-by-play data for {len(game_ids)} games from {team_season_dir}")
        
        for game_id in game_ids:
            filepath = os.path.join(team_season_dir, f"{game_id}.csv")
            
            if not os.path.exists(filepath):
                self.logger.warning(f"File not found: {filepath}")
                continue
            
            df = self.load_game_data(filepath)
            if df is not None:
                games_data.append(df)
        
        self.logger.info(f"Successfully loaded {len(games_data)} games")
        return games_data
    
    def clean_time_string(self, time_str: str) -> int:
        """
        Convert PCTIMESTRING to seconds remaining in game.
        
        Args:
            time_str: Time string in format "MM:SS" or "M:SS"
            
        Returns:
            Seconds remaining in the period
        """
        if pd.isna(time_str) or time_str == '':
            return 0
        
        try:
            # Handle different time formats
            if ':' in str(time_str):
                minutes, seconds = str(time_str).split(':')
                return int(minutes) * 60 + int(seconds)
            else:
                return int(time_str)
        except (ValueError, AttributeError):
            return 0
    
    def parse_score_margin(self, score_str: str, team_abbrev: str) -> int:
        """
        Parse score margin string to get margin from team's perspective.
        
        Args:
            score_str: Score string like "LAL 88 – BOS 90"
            team_abbrev: Team abbreviation (e.g., "IND")
            
        Returns:
            Score margin from team's perspective (positive = leading)
        """
        if pd.isna(score_str) or score_str == '':
            return 0
        
        try:
            # Extract scores using regex
            # Pattern: TEAM1 SCORE1 – TEAM2 SCORE2
            pattern = r'([A-Z]{3})\s+(\d+)\s*–\s*([A-Z]{3})\s+(\d+)'
            match = re.search(pattern, str(score_str))
            
            if match:
                team1, score1, team2, score2 = match.groups()
                score1, score2 = int(score1), int(score2)
                
                # Determine which team is the target team
                if team1 == team_abbrev:
                    return score1 - score2
                elif team2 == team_abbrev:
                    return score2 - score1
                else:
                    return 0
            else:
                return 0
                
        except (ValueError, AttributeError):
            return 0
    
    def categorize_event(self, event_type: int, action_type: int) -> str:
        """
        Categorize event type into simplified categories.
        
        Args:
            event_type: EVENTMSGTYPE
            action_type: EVENTMSGACTIONTYPE
            
        Returns:
            Event category string
        """
        if pd.isna(event_type):
            return 'UNKNOWN'
        
        event_type = int(event_type)
        
        # NBA event type mappings
        event_categories = {
            1: 'SHOT',      # Made Shot
            2: 'SHOT',      # Missed Shot
            3: 'FREE_THROW', # Free Throw
            4: 'REBOUND',   # Rebound
            5: 'TURNOVER',  # Turnover
            6: 'FOUL',      # Foul
            7: 'SUBSTITUTION', # Substitution
            8: 'TIMEOUT',   # Timeout
            9: 'JUMP_BALL', # Jump Ball
            10: 'EJECTION', # Ejection
            11: 'START_PERIOD', # Start of Period
            12: 'END_PERIOD',   # End of Period
            13: 'INSTANT_REPLAY' # Instant Replay
        }
        
        return event_categories.get(event_type, 'UNKNOWN')
    
    def clean_game_data(self, df: pd.DataFrame, team_abbrev: str) -> pd.DataFrame:
        """
        Clean and standardize a single game's play-by-play data.
        
        Args:
            df: Raw play-by-play DataFrame
            team_abbrev: Team abbreviation for score margin calculation
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Ensure core columns exist
        for col in self.core_columns:
            if col not in cleaned_df.columns:
                cleaned_df[col] = None
        
        # Clean time remaining
        cleaned_df['time_remaining_sec'] = cleaned_df['PCTIMESTRING'].apply(
            self.clean_time_string
        )
        
        # Parse score margin
        cleaned_df['score_margin_int'] = cleaned_df['SCOREMARGIN'].apply(
            lambda x: self.parse_score_margin(x, team_abbrev)
        )
        
        # Categorize events
        cleaned_df['event_category'] = cleaned_df.apply(
            lambda row: self.categorize_event(row['EVENTMSGTYPE'], row['EVENTMSGACTIONTYPE']),
            axis=1
        )
        
        # Determine if team is home
        cleaned_df['is_home'] = (
            cleaned_df['PLAYER1_TEAM_ABBREVIATION'] == team_abbrev
        ).fillna(False)
        
        # Add game-level metadata
        cleaned_df['game_period'] = cleaned_df['PERIOD']
        cleaned_df['event_number'] = cleaned_df['EVENTNUM']
        
        # Clean text descriptions
        cleaned_df['home_description'] = cleaned_df['HOMEDESCRIPTION'].fillna('')
        cleaned_df['visitor_description'] = cleaned_df['VISITORDESCRIPTION'].fillna('')
        
        # Clean player names
        cleaned_df['player1_name'] = cleaned_df['PLAYER1_NAME'].fillna('')
        cleaned_df['player2_name'] = cleaned_df['PLAYER2_NAME'].fillna('')
        
        # Select and reorder columns
        output_columns = [
            'GAME_ID', 'event_number', 'game_period', 'time_remaining_sec',
            'score_margin_int', 'event_category', 'is_home',
            'home_description', 'visitor_description',
            'player1_name', 'PLAYER1_TEAM_ABBREVIATION', 'player2_name',
            'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE', 'PERSON1TYPE',
            'TEAM_ID_HOME', 'TEAM_ID_AWAY'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in output_columns if col in cleaned_df.columns]
        cleaned_df = cleaned_df[available_columns]
        
        return cleaned_df
    
    def merge_all_games(self, games_data: List[pd.DataFrame], 
                       team_abbrev: str) -> pd.DataFrame:
        """
        Merge all game data into a single DataFrame.
        
        Args:
            games_data: List of DataFrames with game data
            team_abbrev: Team abbreviation for processing
            
        Returns:
            Merged DataFrame with all games
        """
        if not games_data:
            self.logger.warning("No game data to merge")
            return pd.DataFrame()
        
        self.logger.info(f"Merging {len(games_data)} games")
        
        # Clean each game's data
        cleaned_games = []
        for i, df in enumerate(games_data):
            self.logger.info(f"Cleaning game {i+1}/{len(games_data)}")
            cleaned_df = self.clean_game_data(df, team_abbrev)
            cleaned_games.append(cleaned_df)
        
        # Concatenate all games
        merged_df = pd.concat(cleaned_games, ignore_index=True)
        
        # Sort by game ID and event number
        merged_df = merged_df.sort_values(['GAME_ID', 'event_number']).reset_index(drop=True)
        
        self.logger.info(f"Merged data: {len(merged_df)} total plays across {len(games_data)} games")
        
        return merged_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save processed DataFrame to CSV.
        
        Args:
            df: Processed DataFrame
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if df.empty:
            self.logger.warning("Cannot save empty DataFrame")
            return ""
        
        filepath = os.path.join(self.processed_dir, filename)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Saved {len(df)} plays to {filepath}")
        return filepath
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for processed data.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {
                'total_plays': 0,
                'total_games': 0,
                'avg_plays_per_game': 0,
                'event_categories': {},
                'periods': {}
            }
        
        unique_games = df['GAME_ID'].nunique()
        event_counts = df['event_category'].value_counts().to_dict()
        period_counts = df['game_period'].value_counts().to_dict()
        
        return {
            'total_plays': len(df),
            'total_games': unique_games,
            'avg_plays_per_game': len(df) / unique_games if unique_games > 0 else 0,
            'event_categories': event_counts,
            'periods': period_counts,
            'score_margin_stats': {
                'mean': df['score_margin_int'].mean(),
                'std': df['score_margin_int'].std(),
                'min': df['score_margin_int'].min(),
                'max': df['score_margin_int'].max()
            }
        }
