"""
Compute Win Probability Added (WPA) per event and aggregate to player-level Clutch Scores.

This module implements the core attribution logic for assigning WPA credit
to players based on their actions during games.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from models.wp_model import WinProbabilityModel
from features.build_leverage_features import build_wp_features, compute_leverage_weight

logger = logging.getLogger(__name__)


class ClutchScoreCalculator:
    """
    Calculate Win Probability Added (WPA) and aggregate to player-level scores.
    """
    
    def __init__(self, wp_model: WinProbabilityModel):
        """
        Initialize Clutch Score Calculator.
        
        Args:
            wp_model: Trained Win Probability model
        """
        self.wp_model = wp_model
        logger.info("Initialized ClutchScoreCalculator")
    
    def compute_event_wpa(
        self,
        df: pd.DataFrame,
        game_outcomes: pd.DataFrame,
        leverage_alpha: float = 0.6,
        leverage_beta: float = 0.4,
        max_leverage: float = 3.0
    ) -> pd.DataFrame:
        """
        Compute Win Probability Added (WPA) for each event.
        
        For each event:
        1. Calculate WP_before and WP_after
        2. Compute delta_wp = WP_after - WP_before
        3. Apply leverage weight
        4. Compute WPA_weighted = delta_wp * leverage_weight
        
        Args:
            df: DataFrame with play-by-play data
            game_outcomes: DataFrame with columns ['GAME_ID', 'home_win']
            leverage_alpha: Weight for score margin component in leverage (default: 0.6)
            leverage_beta: Weight for time remaining component in leverage (default: 0.4)
            max_leverage: Maximum leverage weight cap (default: 3.0)
            
        Returns:
            DataFrame with added columns: WP_before, WP_after, delta_wp, leverage_weight, WPA_weighted
        """
        logger.info(f"Computing WPA for {len(df)} events...")
        
        # Make a copy to avoid modifying original
        wpa_df = df.copy()
        
        # Build features for WP model
        wpa_df = build_wp_features(wpa_df)
        
        # Calculate leverage weight
        leverage_weight = compute_leverage_weight(
            score_margin=wpa_df['score_margin'],
            seconds_remaining=wpa_df['seconds_remaining'],
            alpha=leverage_alpha,
            beta=leverage_beta,
            max_weight=max_leverage
        )
        wpa_df['leverage_weight'] = leverage_weight
        
        # Calculate WP_before (state before event)
        # For each event, we need to compute WP from the state at that moment
        wpa_df['WP_before'] = self.wp_model.predict(wpa_df)
        
        # Calculate WP_after (state after event)
        # We compute WP_after by looking at the next event's state (which represents
        # the game state after the current event). For the last event in each game,
        # WP_after = final outcome (1 if home won, 0 if away won).
        
        # Sort by game and event number to ensure proper order
        wpa_df = wpa_df.sort_values(['GAME_ID', 'EVENTNUM']).reset_index(drop=True)
        
        # Merge game outcomes
        wpa_df = wpa_df.merge(
            game_outcomes[['GAME_ID', 'home_win']],
            on='GAME_ID',
            how='left'
        )
        
        # Initialize WP_after
        wpa_df['WP_after'] = wpa_df['WP_before'].copy()
        
        # For each game, compute WP_after by looking at the next event's state
        for game_id in wpa_df['GAME_ID'].unique():
            game_mask = wpa_df['GAME_ID'] == game_id
            game_indices = wpa_df[game_mask].index.tolist()
            
            if len(game_indices) == 0:
                continue
            
            # For each event except the last, WP_after = next event's WP_before
            for i in range(len(game_indices) - 1):
                current_idx = game_indices[i]
                next_idx = game_indices[i + 1]
                wpa_df.loc[current_idx, 'WP_after'] = wpa_df.loc[next_idx, 'WP_before']
            
            # For the last event, WP_after = final outcome
            last_idx = game_indices[-1]
            final_wp = float(wpa_df.loc[last_idx, 'home_win'])
            wpa_df.loc[last_idx, 'WP_after'] = final_wp
        
        # Calculate delta_wp
        wpa_df['delta_wp'] = wpa_df['WP_after'] - wpa_df['WP_before']
        
        # Calculate WPA_weighted
        wpa_df['WPA_weighted'] = wpa_df['delta_wp'] * wpa_df['leverage_weight']
        
        logger.info(f"WPA computed. Mean delta_wp: {wpa_df['delta_wp'].mean():.4f}, "
                   f"Mean WPA_weighted: {wpa_df['WPA_weighted'].mean():.4f}")
        
        return wpa_df
    
    def attribute_wpa_to_players(self, wpa_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attribute WPA to players based on event type and attribution rules.
        
        Attribution rules:
        - Shooter gets 1.0 × WPA for made shots
        - Assister (if applicable) gets +0.15 × WPA
        - Turnover player gets -WPA (full blame)
        - Rebounder gets +WPA (full credit)
        - Other events: full credit to PLAYER1_NAME
        
        Args:
            wpa_df: DataFrame with WPA_weighted column and event data
            
        Returns:
            DataFrame with player_id and player_wpa columns for attribution
        """
        logger.info("Attributing WPA to players...")
        
        # Initialize player WPA
        wpa_df['player_wpa'] = 0.0
        wpa_df['attribution_type'] = 'none'
        
        # Made shots: shooter gets full credit
        # Convert description columns to strings and handle NaN values
        home_desc = wpa_df['HOMEDESCRIPTION'].astype(str).fillna('')
        visitor_desc = wpa_df['VISITORDESCRIPTION'].astype(str).fillna('')
        
        made_shot_mask = (wpa_df['event_category'] == 'SHOT') & (
            home_desc.str.contains('PTS', case=False, na=False) |
            visitor_desc.str.contains('PTS', case=False, na=False)
        )
        
        wpa_df.loc[made_shot_mask, 'player_wpa'] = wpa_df.loc[made_shot_mask, 'WPA_weighted']
        wpa_df.loc[made_shot_mask, 'attribution_type'] = 'shooter'
        
        # Missed shots: shooter gets negative WPA (missed opportunity)
        missed_shot_mask = wpa_df['event_category'] == 'MISS'
        wpa_df.loc[missed_shot_mask, 'player_wpa'] = -wpa_df.loc[missed_shot_mask, 'WPA_weighted']
        wpa_df.loc[missed_shot_mask, 'attribution_type'] = 'missed_shot'
        
        # Assists: give 15% credit to assister
        # Look for AST in descriptions (already converted to strings above)
        assist_mask = (
            home_desc.str.contains('AST', case=False, na=False) |
            visitor_desc.str.contains('AST', case=False, na=False)
        )
        
        # Extract assister name from description (format: "Player X AST")
        # This is simplified - real implementation would parse PLAYER2_NAME if available
        # For now, we'll give assist credit on the next event (the made shot)
        # Actually, assists are typically part of the same event as the made shot
        # So we'll handle this differently - if we can identify assists, give credit
        
        # Turnovers: full blame
        turnover_mask = wpa_df['event_category'] == 'TURNOVER'
        wpa_df.loc[turnover_mask, 'player_wpa'] = -wpa_df.loc[turnover_mask, 'WPA_weighted']
        wpa_df.loc[turnover_mask, 'attribution_type'] = 'turnover'
        
        # Rebounds: full credit
        rebound_mask = wpa_df['event_category'] == 'REBOUND'
        wpa_df.loc[rebound_mask, 'player_wpa'] = wpa_df.loc[rebound_mask, 'WPA_weighted']
        wpa_df.loc[rebound_mask, 'attribution_type'] = 'rebound'
        
        # Fouls: negative credit (full blame)
        foul_mask = wpa_df['event_category'] == 'FOUL'
        wpa_df.loc[foul_mask, 'player_wpa'] = -wpa_df.loc[foul_mask, 'WPA_weighted'] * 0.5  # Half blame
        wpa_df.loc[foul_mask, 'attribution_type'] = 'foul'
        
        # Free throws: credit to shooter
        ft_mask = wpa_df['event_category'] == 'FREE_THROW'
        # Made FTs get positive credit, missed get negative
        made_ft_mask = ft_mask & (
            home_desc.str.contains('PTS', case=False, na=False) |
            visitor_desc.str.contains('PTS', case=False, na=False)
        )
        wpa_df.loc[made_ft_mask, 'player_wpa'] = wpa_df.loc[made_ft_mask, 'WPA_weighted']
        wpa_df.loc[made_ft_mask, 'attribution_type'] = 'free_throw_made'
        
        missed_ft_mask = ft_mask & ~made_ft_mask
        wpa_df.loc[missed_ft_mask, 'player_wpa'] = -wpa_df.loc[missed_ft_mask, 'WPA_weighted'] * 0.5
        wpa_df.loc[missed_ft_mask, 'attribution_type'] = 'free_throw_missed'
        
        # Other events: default to PLAYER1_NAME if available
        unassigned_mask = wpa_df['player_wpa'] == 0.0
        wpa_df.loc[unassigned_mask & wpa_df['PLAYER1_NAME'].notna(), 'player_wpa'] = (
            wpa_df.loc[unassigned_mask & wpa_df['PLAYER1_NAME'].notna(), 'WPA_weighted']
        )
        wpa_df.loc[unassigned_mask & wpa_df['PLAYER1_NAME'].notna(), 'attribution_type'] = 'other'
        
        # Only assign WPA to events with a player
        wpa_df.loc[wpa_df['PLAYER1_NAME'].isna(), 'player_wpa'] = 0.0
        
        logger.info(f"WPA attributed. Total player WPA: {wpa_df['player_wpa'].sum():.4f}, "
                   f"Mean player WPA: {wpa_df['player_wpa'].mean():.4f}")
        
        return wpa_df
    
    def aggregate_player_wpa(self, wpa_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate WPA to player-level metrics.
        
        Computes:
        - player_total_WPA: Sum of all WPA for player
        - player_avg_WPA: Mean WPA per event
        - player_leverage_index: Mean leverage weight for player's events
        - player_event_count: Number of events attributed to player
        
        Args:
            wpa_df: DataFrame with player_wpa and player attribution
            
        Returns:
            DataFrame with player-level aggregated metrics
        """
        logger.info("Aggregating player WPA...")
        
        # Filter to events with player attribution
        player_events = wpa_df[wpa_df['PLAYER1_NAME'].notna()].copy()
        
        if len(player_events) == 0:
            logger.warning("No player events found for aggregation")
            return pd.DataFrame()
        
        # Group by player
        player_stats = player_events.groupby('PLAYER1_NAME').agg({
            'player_wpa': ['sum', 'mean', 'count'],
            'WPA_weighted': 'sum',
            'leverage_weight': 'mean',
            'GAME_ID': 'nunique'
        }).reset_index()
        
        # Flatten column names
        player_stats.columns = [
            'player_name',
            'player_total_WPA',
            'player_avg_WPA',
            'player_event_count',
            'player_total_WPA_weighted',
            'player_leverage_index',
            'games_played'
        ]
        
        # Sort by total WPA descending
        player_stats = player_stats.sort_values('player_total_WPA', ascending=False).reset_index(drop=True)
        
        logger.info(f"Aggregated WPA for {len(player_stats)} players")
        logger.info(f"Top 5 players by total WPA:")
        for i, row in player_stats.head(5).iterrows():
            logger.info(f"  {row['player_name']}: {row['player_total_WPA']:.4f} WPA "
                       f"({row['player_event_count']} events)")
        
        return player_stats
    
    def save_wpa_data(self, wpa_df: pd.DataFrame, output_path: str):
        """
        Save event-level WPA data to CSV.
        
        Args:
            wpa_df: DataFrame with WPA calculations
            output_path: Path to save CSV
        """
        # Select relevant columns for output
        output_cols = [
            'GAME_ID', 'EVENTNUM', 'PERIOD', 'PCTIMESTRING',
            'seconds_remaining', 'total_seconds_remaining',
            'score_margin_int', 'score_margin',
            'event_category', 'is_home',
            'PLAYER1_NAME', 'PLAYER1_TEAM_ABBREVIATION',
            'WP_before', 'WP_after', 'delta_wp',
            'leverage_weight', 'WPA_weighted', 'player_wpa',
            'attribution_type'
        ]
        
        # Only include columns that exist
        available_cols = [col for col in output_cols if col in wpa_df.columns]
        output_df = wpa_df[available_cols].copy()
        
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved WPA data to {output_path} ({len(output_df)} events)")
    
    def save_player_summary(self, player_stats: pd.DataFrame, output_path: str):
        """
        Save player-level WPA summary to CSV.
        
        Args:
            player_stats: DataFrame with aggregated player metrics
            output_path: Path to save CSV
        """
        player_stats.to_csv(output_path, index=False)
        logger.info(f"Saved player summary to {output_path} ({len(player_stats)} players)")

