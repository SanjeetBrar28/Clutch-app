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
        Convert PCTIMESTRING to seconds remaining in period.
        Handles formats like "PT12M00.00S" or "05:32"
        
        Args:
            time_str: Time string in various formats
            
        Returns:
            Seconds remaining in the period
        """
        if pd.isna(time_str) or time_str == '':
            return 0
        
        try:
            time_str = str(time_str).strip()
            
            # Handle ISO 8601 duration format: PT12M00.00S
            if time_str.startswith('PT'):
                # Remove PT prefix and S suffix
                time_str = time_str.replace('PT', '').replace('S', '')
                # Split by M
                if 'M' in time_str:
                    parts = time_str.split('M')
                    minutes = float(parts[0]) if parts[0] else 0
                    seconds = float(parts[1]) if len(parts) > 1 and parts[1] else 0
                    return int(minutes * 60 + seconds)
                else:
                    return int(float(time_str))
            
            # Handle MM:SS format
            elif ':' in time_str:
                parts = time_str.split(':')
                minutes = int(parts[0])
                seconds = float(parts[1])
                return int(minutes * 60 + seconds)
            else:
                return int(float(time_str))
        except (ValueError, AttributeError):
            return 0
    
    def parse_score_margin(self, score_home: float, score_away: float, 
                          team_is_home: bool, team_abbrev: str) -> int:
        """
        Calculate score margin from team's perspective using score columns.
        
        Args:
            score_home: Home team score
            score_away: Away team score
            team_is_home: Whether the team is home
            team_abbrev: Team abbreviation (for fallback if needed)
            
        Returns:
            Score margin from team's perspective (positive = leading)
        """
        try:
            score_home = float(score_home) if pd.notna(score_home) else 0
            score_away = float(score_away) if pd.notna(score_away) else 0
            
            if team_is_home:
                return int(score_home - score_away)
            else:
                return int(score_away - score_home)
        except (ValueError, TypeError):
            return 0
    
    def normalize_event_category(self, event_type: str, event_action_type: str = None) -> str:
        """
        Normalize event type into standardized categories.
        
        Args:
            event_type: EVENTMSGTYPE (can be string like "Made Shot", "Missed Shot", etc.)
            event_action_type: EVENTMSGACTIONTYPE (subtype like "Pullup Jump Shot")
            
        Returns:
            Normalized event category: SHOT, MISS, FREE_THROW, REBOUND, FOUL, TURNOVER, SUBSTITUTION, OTHER
        """
        if pd.isna(event_type) or event_type == '':
            return 'OTHER'
        
        cat = str(event_type).strip().upper()
        action_cat = str(event_action_type).strip().upper() if pd.notna(event_action_type) and event_action_type != '' else ''
        
        # Check both event type and action type for better categorization
        combined = f"{cat} {action_cat}"
        
        # Normalize to standardized set
        if "SHOT" in combined or "MADE" in combined:
            if "FREE" in combined or "FREE THROW" in combined:
                return "FREE_THROW"
            return "SHOT"
        elif "MISS" in combined and "FREE" not in combined:
            return "MISS"
        elif "FREE" in combined or "FREE THROW" in combined:
            return "FREE_THROW"
        elif "FOUL" in combined:
            return "FOUL"
        elif "REBOUND" in combined:
            return "REBOUND"
        elif "TURNOVER" in combined:
            return "TURNOVER"
        elif "SUBSTITUTION" in combined or "SUB" in combined:
            return "SUBSTITUTION"
        else:
            return "OTHER"
    
    def _get_column(self, df: pd.DataFrame, possible_names: List[str]):
        """Get column by case-insensitive matching."""
        df_cols_lower = {col.lower(): col for col in df.columns}
        for name in possible_names:
            if name.lower() in df_cols_lower:
                return df_cols_lower[name.lower()]
        return None
    
    def clean_game_data(self, df: pd.DataFrame, team_abbrev: str) -> pd.DataFrame:
        """
        Clean and standardize a single game's play-by-play data.
        
        Args:
            df: Raw play-by-play DataFrame
            team_abbrev: Team abbreviation for score margin calculation and home/away detection
            
        Returns:
            Cleaned DataFrame with all required columns and derived features
        """
        if df.empty:
            return df
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Normalize column names (case-insensitive matching)
        col_map = {}
        
        # Map essential columns with case-insensitive matching
        essential_cols = {
            'GAME_ID': ['GAME_ID', 'gameId', 'game_id'],
            'EVENTNUM': ['EVENTNUM', 'actionNumber', 'eventnum', 'EVENT_NUM'],
            'PERIOD': ['PERIOD', 'period'],
            'PCTIMESTRING': ['PCTIMESTRING', 'clock', 'pctimestring', 'PCTIME'],
            'HOMEDESCRIPTION': ['HOMEDESCRIPTION', 'description', 'homedescription'],
            'VISITORDESCRIPTION': ['VISITORDESCRIPTION', 'visitordescription'],
            'PLAYER1_NAME': ['PLAYER1_NAME', 'playerName', 'player_name'],
            'PLAYER1_TEAM_ABBREVIATION': ['PLAYER1_TEAM_ABBREVIATION', 'teamTricode', 'team_abbreviation'],
            'EVENTMSGTYPE': ['EVENTMSGTYPE', 'actionType', 'eventmsgtype'],
            'EVENTMSGACTIONTYPE': ['EVENTMSGACTIONTYPE', 'subType', 'eventmsgactiontype'],
            'TEAM_ID_HOME': ['TEAM_ID_HOME', 'teamId', 'team_id_home'],
            'TEAM_ID_AWAY': ['TEAM_ID_AWAY', 'team_id_away'],
            'SCORE_HOME': ['HOME_SCORE', 'scoreHome', 'score_home', 'home_score'],
            'SCORE_AWAY': ['AWAY_SCORE', 'scoreAway', 'score_away', 'away_score'],
            'LOCATION': ['location', 'LOCATION']
        }
        
        for standard_name, possible_names in essential_cols.items():
            col_name = self._get_column(cleaned_df, possible_names)
            if col_name:
                if col_name != standard_name:
                    cleaned_df[standard_name] = cleaned_df[col_name]
                col_map[standard_name] = col_name
        
        # Fill missing columns with None
        for col in essential_cols.keys():
            if col not in cleaned_df.columns:
                cleaned_df[col] = None
        
        # Normalize missing values (empty strings → NaN)
        text_cols = ['HOMEDESCRIPTION', 'VISITORDESCRIPTION', 'PLAYER1_NAME', 
                     'PLAYER1_TEAM_ABBREVIATION', 'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE']
        for col in text_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].replace('', pd.NA)
                cleaned_df[col] = cleaned_df[col].replace(' ', pd.NA)
        
        # Determine if team is home (check location column or team IDs)
        if 'LOCATION' in cleaned_df.columns:
            cleaned_df['is_home'] = (cleaned_df['LOCATION'].str.lower() == 'h').fillna(False)
        else:
            # Fallback: check if team abbreviation matches home team indicator
            cleaned_df['is_home'] = False  # Will be set properly below
        
        # Derive: seconds_remaining_period
        cleaned_df['seconds_remaining_period'] = cleaned_df['PCTIMESTRING'].apply(
            self.clean_time_string
        )
        
        # Derive: total_seconds_remaining = (4-period)*720 + seconds_remaining_period
        # Handle regular periods (1-4) and overtime (5+)
        def calc_total_seconds(row):
            period = int(row['PERIOD']) if pd.notna(row['PERIOD']) else 4
            secs_in_period = int(row['seconds_remaining_period']) if pd.notna(row['seconds_remaining_period']) else 0
            
            if period <= 4:
                # Regular game: (4 - period) * 720 + seconds_remaining
                return (4 - period) * 720 + secs_in_period
            else:
                # Overtime: only seconds remaining in OT period
                return secs_in_period
        
        cleaned_df['total_seconds_remaining'] = cleaned_df.apply(calc_total_seconds, axis=1).astype(int)
        
        # Derive: score_margin_int (from team's perspective)
        cleaned_df['score_margin_int'] = cleaned_df.apply(
            lambda row: self.parse_score_margin(
                row.get('SCORE_HOME', 0),
                row.get('SCORE_AWAY', 0),
                row.get('is_home', False),
                team_abbrev
            ),
            axis=1
        )
        
        # Derive: event_category with normalization
        cleaned_df['event_category'] = cleaned_df.apply(
            lambda row: self.normalize_event_category(
                row.get('EVENTMSGTYPE', ''),
                row.get('EVENTMSGACTIONTYPE', '')
            ),
            axis=1
        )
        
        # Determine is_home more accurately using team abbreviation in data
        if 'PLAYER1_TEAM_ABBREVIATION' in cleaned_df.columns:
            # Check if any play has the team as home
            has_team_plays = cleaned_df['PLAYER1_TEAM_ABBREVIATION'] == team_abbrev
            if has_team_plays.any():
                # Get location from a play where this team was involved
                team_play = cleaned_df[has_team_plays].iloc[0] if has_team_plays.any() else None
                if team_play is not None and 'LOCATION' in cleaned_df.columns:
                    cleaned_df['is_home'] = (cleaned_df['LOCATION'].str.lower() == 'h').fillna(False)
        
        # Filter out invalid events (Start Period, End Period, etc.)
        events_to_filter = ['OTHER']
        cleaned_df = cleaned_df[~cleaned_df['event_category'].isin(events_to_filter)]
        
        # Additional filtering for period start/end events
        if 'HOMEDESCRIPTION' in cleaned_df.columns:
            period_events = cleaned_df['HOMEDESCRIPTION'].str.contains(
                'Start of|End of|period', case=False, na=False
            )
            cleaned_df = cleaned_df[~period_events]
        
        # Drop empty or meaningless rows
        # Remove rows with no meaningful content
        before_drop = len(cleaned_df)
        
        # Drop rows where all key fields are empty
        key_fields = ['PLAYER1_NAME', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION']
        has_content = pd.Series([False] * len(cleaned_df))
        
        for field in key_fields:
            if field in cleaned_df.columns:
                has_content = has_content | cleaned_df[field].notna()
        
        cleaned_df = cleaned_df[has_content]
        
        rows_dropped = before_drop - len(cleaned_df)
        if rows_dropped > 0:
            self.logger.info(f"Dropped {rows_dropped} empty/meaningless rows")
        
        # Forward-fill team IDs (fill missing values from previous row within same game)
        if 'TEAM_ID_HOME' in cleaned_df.columns:
            cleaned_df['TEAM_ID_HOME'] = cleaned_df.groupby('GAME_ID')['TEAM_ID_HOME'].ffill()
        if 'TEAM_ID_AWAY' in cleaned_df.columns:
            cleaned_df['TEAM_ID_AWAY'] = cleaned_df.groupby('GAME_ID')['TEAM_ID_AWAY'].ffill()
        
        # Ensure proper data types with safe coercion and default to 0
        if 'PERIOD' in cleaned_df.columns:
            cleaned_df['PERIOD'] = pd.to_numeric(cleaned_df['PERIOD'], errors='coerce').fillna(0).astype(int)
        if 'EVENTNUM' in cleaned_df.columns:
            cleaned_df['EVENTNUM'] = pd.to_numeric(cleaned_df['EVENTNUM'], errors='coerce').fillna(0).astype(int)
        
        # Ensure numeric fields are properly typed as integers with 0 for missing
        if 'score_margin_int' in cleaned_df.columns:
            cleaned_df['score_margin_int'] = pd.to_numeric(
                cleaned_df['score_margin_int'], errors='coerce'
            ).fillna(0).astype(int)
        
        if 'seconds_remaining_period' in cleaned_df.columns:
            cleaned_df['seconds_remaining_period'] = pd.to_numeric(
                cleaned_df['seconds_remaining_period'], errors='coerce'
            ).fillna(0).astype(int)
        
        if 'total_seconds_remaining' in cleaned_df.columns:
            cleaned_df['total_seconds_remaining'] = pd.to_numeric(
                cleaned_df['total_seconds_remaining'], errors='coerce'
            ).fillna(0).astype(int)
        
        # Select and order final columns
        output_columns = [
            'GAME_ID',
            'EVENTNUM',
            'PERIOD',
            'PCTIMESTRING',
            'seconds_remaining_period',
            'total_seconds_remaining',
            'score_margin_int',
            'event_category',
            'is_home',
            'HOMEDESCRIPTION',
            'VISITORDESCRIPTION',
            'PLAYER1_NAME',
            'PLAYER1_TEAM_ABBREVIATION',
            'EVENTMSGTYPE',
            'EVENTMSGACTIONTYPE',
            'TEAM_ID_HOME',
            'TEAM_ID_AWAY'
        ]
        
        # Only include columns that exist and have data
        available_columns = [col for col in output_columns if col in cleaned_df.columns]
        cleaned_df = cleaned_df[available_columns]
        
        # Reset index
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        return cleaned_df
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate processed data and return summary statistics.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(df),
            'unique_games': df['GAME_ID'].nunique() if 'GAME_ID' in df.columns else 0,
            'event_category_distribution': {},
            'missing_field_counts': {},
            'numeric_field_stats': {},
            'validation_passed': True
        }
        
        if df.empty:
            validation_results['validation_passed'] = False
            return validation_results
        
        # Event category distribution
        if 'event_category' in df.columns:
            validation_results['event_category_distribution'] = df['event_category'].value_counts().to_dict()
        
        # Missing field counts
        key_fields = ['GAME_ID', 'EVENTNUM', 'PERIOD', 'PLAYER1_NAME', 
                      'HOMEDESCRIPTION', 'VISITORDESCRIPTION', 'TEAM_ID_HOME', 'TEAM_ID_AWAY']
        for field in key_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                validation_results['missing_field_counts'][field] = int(missing_count)
        
        # Numeric field statistics
        numeric_fields = ['score_margin_int', 'seconds_remaining_period', 
                         'total_seconds_remaining', 'PERIOD']
        for field in numeric_fields:
            if field in df.columns:
                stats = {
                    'min': float(df[field].min()) if len(df) > 0 else 0,
                    'max': float(df[field].max()) if len(df) > 0 else 0,
                    'mean': float(df[field].mean()) if len(df) > 0 else 0,
                    'null_count': int(df[field].isna().sum())
                }
                validation_results['numeric_field_stats'][field] = stats
        
        return validation_results
    
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
        
        # Apply forward-fill for team IDs across all games (in case some games are missing)
        if 'TEAM_ID_HOME' in merged_df.columns:
            merged_df['TEAM_ID_HOME'] = merged_df.groupby('GAME_ID')['TEAM_ID_HOME'].ffill()
        if 'TEAM_ID_AWAY' in merged_df.columns:
            merged_df['TEAM_ID_AWAY'] = merged_df.groupby('GAME_ID')['TEAM_ID_AWAY'].ffill()
        
        # Sort by game ID and event number
        sort_cols = ['GAME_ID', 'EVENTNUM'] if 'EVENTNUM' in merged_df.columns else ['GAME_ID']
        merged_df = merged_df.sort_values(sort_cols).reset_index(drop=True)
        
        self.logger.info(f"Merged data: {len(merged_df)} total plays across {len(games_data)} games")
        
        return merged_df
    
    def log_validation_results(self, validation_results: Dict, log_file: str = None):
        """
        Log validation results to a dedicated validation log file.
        
        Args:
            validation_results: Dictionary with validation results
            log_file: Path to validation log file (default: data/logs/process_validation.log)
        """
        if log_file is None:
            log_file = os.path.join('data', 'logs', 'process_validation.log')
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Write validation results to log file
        with open(log_file, 'a') as f:
            from datetime import datetime
            f.write(f"\n{'='*60}\n")
            f.write(f"Validation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"✅ Processed {validation_results['total_rows']:,} events across {validation_results['unique_games']} games\n\n")
            
            f.write(f"📊 Event type distribution:\n")
            for category, count in sorted(validation_results['event_category_distribution'].items(), 
                                         key=lambda x: x[1], reverse=True):
                percentage = (count / validation_results['total_rows']) * 100 if validation_results['total_rows'] > 0 else 0
                f.write(f"   {category}: {count:,} ({percentage:.2f}%)\n")
            
            f.write(f"\n🔍 Missing field counts:\n")
            for field, count in validation_results['missing_field_counts'].items():
                if count > 0:
                    f.write(f"   {field}: {count:,}\n")
                else:
                    f.write(f"   {field}: 0 ✓\n")
            
            f.write(f"\n📈 Numeric field statistics:\n")
            for field, stats in validation_results['numeric_field_stats'].items():
                f.write(f"   {field}:\n")
                f.write(f"      Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']:.2f}\n")
                f.write(f"      Null count: {stats['null_count']}\n")
            
            f.write(f"\n{'='*60}\n\n")
    
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
            Dictionary with summary statistics including games, rows, duration, missing values
        """
        if df.empty:
            return {
                'total_plays': 0,
                'total_games': 0,
                'avg_plays_per_game': 0,
                'event_categories': {},
                'periods': {},
                'missing_values': {},
                'duration_info': {}
            }
        
        unique_games = df['GAME_ID'].nunique() if 'GAME_ID' in df.columns else 0
        event_counts = df['event_category'].value_counts().to_dict() if 'event_category' in df.columns else {}
        
        # Period counts
        if 'PERIOD' in df.columns:
            period_counts = df['PERIOD'].value_counts().sort_index().to_dict()
        else:
            period_counts = {}
        
        # Missing values summary
        missing_values = {}
        if 'HOMEDESCRIPTION' in df.columns:
            missing_values['HOMEDESCRIPTION'] = int(df['HOMEDESCRIPTION'].isna().sum())
        if 'PLAYER1_NAME' in df.columns:
            missing_values['PLAYER1_NAME'] = int(df['PLAYER1_NAME'].isna().sum())
        
        # Duration info
        duration_info = {}
        if 'total_seconds_remaining' in df.columns:
            duration_info['min_seconds'] = int(df['total_seconds_remaining'].min())
            duration_info['max_seconds'] = int(df['total_seconds_remaining'].max())
            duration_info['mean_seconds'] = float(df['total_seconds_remaining'].mean())
        
        # Score margin stats
        score_margin_stats = {}
        if 'score_margin_int' in df.columns:
            score_margin_stats = {
                'mean': float(df['score_margin_int'].mean()),
                'std': float(df['score_margin_int'].std()),
                'min': int(df['score_margin_int'].min()),
                'max': int(df['score_margin_int'].max())
            }
        
        return {
            'total_plays': len(df),
            'total_games': unique_games,
            'avg_plays_per_game': len(df) / unique_games if unique_games > 0 else 0,
            'event_categories': event_counts,
            'periods': period_counts,
            'score_margin_stats': score_margin_stats,
            'missing_values': missing_values,
            'duration_info': duration_info
        }
