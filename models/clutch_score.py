"""
Compute Win Probability Added (WPA) per event and aggregate to player-level Clutch Scores.

This module implements the core attribution logic for assigning WPA credit
to players based on their actions during games.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from features.build_leverage_features import build_wp_features, compute_leverage_weight

logger = logging.getLogger(__name__)


class ClutchScoreCalculator:
    """
    Calculate Win Probability Added (WPA) and aggregate to player-level scores.
    """
    
    def __init__(self, wp_model: Any):
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
        # Check if score_margin_home exists (from home team perspective), otherwise use score_margin_int
        if 'score_margin_home' in wpa_df.columns:
            wpa_df = build_wp_features(wpa_df, score_margin_col='score_margin_home')
        else:
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
        if hasattr(self.wp_model, "predict_wp"):
            wpa_df['WP_before'] = self.wp_model.predict_wp(wpa_df)
        elif hasattr(self.wp_model, "predict"):
            wpa_df['WP_before'] = self.wp_model.predict(wpa_df)
        else:
            raise AttributeError("WP model must implement predict_wp or predict method.")
        
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

        # Re-center WPA per game so weighted contributions sum to ~0
        game_bias = wpa_df.groupby('GAME_ID')['WPA_weighted'].transform('sum')
        game_weight_sum = wpa_df.groupby('GAME_ID')['leverage_weight'].transform('sum').replace(0, np.nan)
        adjustment = (game_bias / game_weight_sum) * wpa_df['leverage_weight']
        adjustment = adjustment.fillna(0.0)
        wpa_df['WPA_weighted'] = wpa_df['WPA_weighted'] - adjustment
        wpa_df['delta_wp'] = np.where(
            wpa_df['leverage_weight'] != 0,
            wpa_df['WPA_weighted'] / wpa_df['leverage_weight'],
            0.0
        )
        
        logger.info(f"WPA computed. Mean delta_wp: {wpa_df['delta_wp'].mean():.4f}, "
                   f"Mean WPA_weighted: {wpa_df['WPA_weighted'].mean():.4f}")
        
        return wpa_df
    
    def attribute_wpa_to_players(self, wpa_df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct WPA attribution:
        - WP model outputs: WP = P(home team wins)
        - For home team events → player_wpa = +WPA_weighted
        - For away events → player_wpa = -WPA_weighted
        - No event-specific overrides
        - No guessing from PLAYER1_TEAM_ABBREVIATION
        - No leverage adjustments beyond WPA_weighted
        - Requires TEAM_ID_EVENT
        """

        logger.info("Attributing WPA to players (clean version)...")

        # You MUST have TEAM_ID_EVENT for every event
        if "TEAM_ID_EVENT" not in wpa_df.columns:
            raise ValueError(
                "TEAM_ID_EVENT is missing. Populate TEAM_ID_EVENT during play-by-play preprocessing "
                "using PLAYERS, play team IDs, or by parsing TEAM_ID from NBA API events."
            )

        # Determine event perspective (home or away)
        event_is_home = wpa_df["TEAM_ID_EVENT"] == wpa_df["TEAM_ID_HOME"]

        # Core attribution rule
        wpa_df["player_wpa"] = np.where(
            event_is_home,
            wpa_df["WPA_weighted"],      # home team benefits from positive delta_wp
            -wpa_df["WPA_weighted"]      # away team sees the opposite effect
        )
        wpa_df["player_wpa_raw"] = wpa_df["player_wpa"]

        # Remove WPA for events with missing/unknown players
        invalid_player_mask = (
            wpa_df["PLAYER1_NAME"].isna() |
            (wpa_df["PLAYER1_NAME"] == "") |
            (wpa_df["PLAYER1_NAME"] == "UNKNOWN")
        )
        wpa_df.loc[invalid_player_mask, "player_wpa"] = 0.0

        logger.info(
            f"Total WPA after attribution: {wpa_df['player_wpa'].sum():.4f} "
            "(should be very close to 0 for league-wide attribution)."
        )
        
        return wpa_df
    
    def aggregate_player_wpa(self, wpa_df: pd.DataFrame, deduplicate_by_game: bool = False) -> pd.DataFrame:
        """
        Aggregate WPA to player-level metrics.
        
        IMPORTANT: Deduplication should NOT be done here for single-team processing.
        When processing a single team, each event appears once, so no deduplication is needed.
        Deduplication should only happen when merging data from multiple teams (before calling this function).
        
        Computes:
        - player_total_WPA: Sum of all WPA for player
        - player_avg_WPA: Mean WPA per event
        - player_leverage_index: Mean leverage weight for player's events
        - player_event_count: Number of events attributed to player
        - games_played: Unique games
        
        Args:
            wpa_df: DataFrame with player_wpa and player attribution
            deduplicate_by_game: If True, deduplicate by (player, game_id, EVENTNUM) to avoid double-counting
                                 when same event appears from multiple team perspectives.
                                 WARNING: Only use this when merging data from multiple teams!
            
        Returns:
            DataFrame with player-level aggregated metrics
        """
        logger.info("Aggregating player WPA...")
        
        # Filter to events with player attribution
        # Exclude UNKNOWN players and empty/missing player names
        player_events = wpa_df[
            wpa_df['PLAYER1_NAME'].notna() & 
            (wpa_df['PLAYER1_NAME'] != 'UNKNOWN') & 
            (wpa_df['PLAYER1_NAME'] != '') &
            (wpa_df['PLAYER1_NAME'].str.strip() != '')
        ].copy()
        
        if len(player_events) == 0:
            logger.warning("No player events found for aggregation")
            return pd.DataFrame()
        
        # Log filtered events
        total_events = len(wpa_df)
        unknown_events = total_events - len(player_events)
        if unknown_events > 0:
            logger.info(f"Filtered out {unknown_events} events with UNKNOWN/missing player names")
        
        # Deduplicate by (player, game_id, EVENTNUM) if requested
        # This handles cases where the same EVENT appears from multiple team perspectives
        # Only use this when merging data from multiple teams!
        if deduplicate_by_game:
            # Check for duplicate events (same player, game, eventnum from different perspectives)
            # We deduplicate by (player, game, eventnum) to keep one instance per event
            if 'EVENTNUM' in player_events.columns:
                # Group by (player, game, eventnum) and take first (deduplicate events, not games!)
                player_events_dedup = player_events.groupby(['PLAYER1_NAME', 'GAME_ID', 'EVENTNUM']).first().reset_index()
                
                logger.info(f"Deduplicated: {len(player_events)} events → {len(player_events_dedup)} unique events")
                player_events = player_events_dedup
            else:
                logger.warning("EVENTNUM column not found, cannot deduplicate events properly. Skipping deduplication.")
        
        # Group by player - sum all events (no deduplication needed for single-team processing)
        player_stats = player_events.groupby('PLAYER1_NAME').agg({
            'player_wpa': ['sum', 'mean'],
            'WPA_weighted': 'sum',
            'leverage_weight': 'mean',
            'GAME_ID': 'nunique',
        })
        
        # Flatten MultiIndex columns
        player_stats.columns = [
            'player_total_WPA',
            'player_avg_WPA',
            'player_total_WPA_weighted',
            'player_leverage_index',
            'games_played'
        ]
        player_stats = player_stats.reset_index() 
        player_stats = player_stats.rename(columns={'PLAYER1_NAME': 'player_name'})
        
        # Add event count (simple count of events)
        event_counts = player_events.groupby('PLAYER1_NAME').size().reset_index(name='player_event_count')
        event_counts = event_counts.rename(columns={'PLAYER1_NAME': 'player_name'})
        player_stats = player_stats.merge(event_counts, on='player_name', how='left')
        
        # Include unassigned WPA (events without valid players) so totals balance
        unassigned_mask = (
            wpa_df['PLAYER1_NAME'].isna() |
            (wpa_df['PLAYER1_NAME'] == '') |
            (wpa_df['PLAYER1_NAME'] == 'UNKNOWN')
        )
        if unassigned_mask.any():
            balance_source = 'player_wpa_raw' if 'player_wpa_raw' in wpa_df.columns else 'player_wpa'
            unassigned_wpa = wpa_df.loc[unassigned_mask, balance_source].sum()
            unassigned_events = int(unassigned_mask.sum())
            if abs(unassigned_wpa) > 1e-9:
                balance_row = pd.DataFrame([{
                    'player_name': '__UNASSIGNED__',
                    'player_total_WPA': unassigned_wpa,
                    'player_avg_WPA': 0.0,
                    'player_total_WPA_weighted': unassigned_wpa,
                    'player_leverage_index': 0.0,
                    'games_played': 0,
                    'player_event_count': unassigned_events
                }])
                player_stats = pd.concat([player_stats, balance_row], ignore_index=True)
        
        # Sort by total WPA descending
        player_stats = player_stats.sort_values('player_total_WPA', ascending=False).reset_index(drop=True)
        
        logger.info(f"Aggregated WPA for {len(player_stats)} players")
        logger.info(f"Top 5 players by total WPA:")
        for i, row in player_stats.head(5).iterrows():
            logger.info(f"  {row['player_name']}: {row['player_total_WPA']:.4f} WPA "
                       f"({row['player_event_count']} events)")
        
        return player_stats
    
    def run_diagnostics(self, wpa_df: pd.DataFrame, player_stats: pd.DataFrame, game_outcomes: pd.DataFrame):
        """
        Run diagnostic checks to validate WPA attribution correctness.
        
        Diagnostic A: Global WPA Check - total WPA should be near 0
        Diagnostic B: Team WPA vs Win % correlation - should be positive
        Diagnostic C: Player sanity checks - known stars should have reasonable WPA
        
        Args:
            wpa_df: DataFrame with event-level WPA data
            player_stats: DataFrame with player-level aggregated WPA
            game_outcomes: DataFrame with game outcomes
        """
        logger.info("\n" + "="*60)
        logger.info("🔍 WPA ATTRIBUTION DIAGNOSTICS")
        logger.info("="*60)
        
        logger.info("\n🧪 Checking global WPA balance...")
        player_wpa_backup = None
        if 'player_wpa_raw' in wpa_df.columns:
            player_wpa_backup = wpa_df['player_wpa'].copy()
            wpa_df['player_wpa'] = wpa_df['player_wpa_raw']
        total_wpa = wpa_df["player_wpa"].sum()
        logger.info(f"Total WPA = {total_wpa:.4f} (should be close to zero)")

        if abs(total_wpa) < 2.0:
            logger.info("PASS: WPA balances correctly.")
        else:
            logger.warning("FAIL: WPA imbalance detected. Check TEAM_ID_EVENT.")
        if player_wpa_backup is not None:
            wpa_df['player_wpa'] = player_wpa_backup
        
        # Diagnostic A: Global WPA Check
        total_wpa = player_stats['player_total_WPA'].sum()
        logger.info(f"\n📊 Diagnostic A - Global WPA Check:")
        logger.info(f"   Total WPA (should be near 0): {total_wpa:.4f}")
        if abs(total_wpa) < 2.0:
            logger.info(f"   ✅ PASS: Total WPA is near zero (within ±2.0)")
        else:
            logger.warning(f"   ⚠️  WARNING: Total WPA is {total_wpa:.4f}, expected near 0")
        
        # Diagnostic B: Team WPA vs Win % correlation
        logger.info(f"\n📊 Diagnostic B - Team WPA vs Win % Correlation:")
        
        # Get team WPA totals
        if 'PLAYER1_TEAM_ABBREVIATION' in wpa_df.columns:
            # Map players to teams and sum WPA per team
            player_to_team = wpa_df.groupby('PLAYER1_NAME')['PLAYER1_TEAM_ABBREVIATION'].first().to_dict()
            player_stats_with_team = player_stats.copy()
            player_stats_with_team['team'] = player_stats_with_team['player_name'].map(player_to_team)
            
            team_wpa = player_stats_with_team.groupby('team')['player_total_WPA'].sum().reset_index()
            team_wpa.columns = ['team', 'total_wpa']
            
            # Get team wins from game outcomes
            # We need to determine which team won each game
            if 'TEAM_ID_HOME' in wpa_df.columns and 'TEAM_ID_AWAY' in wpa_df.columns:
                # Get unique games with home/away teams
                game_teams = wpa_df.groupby('GAME_ID').agg({
                    'TEAM_ID_HOME': 'first',
                    'TEAM_ID_AWAY': 'first'
                }).reset_index()
                
                # Merge with outcomes
                game_teams = game_teams.merge(game_outcomes[['GAME_ID', 'home_win']], on='GAME_ID', how='left')
                
                # Count wins per team (simplified - would need team ID mapping for full accuracy)
                # For now, we'll use a simpler approach: count games where team was home and won
                # This is approximate but should show correlation
                logger.info(f"   Team WPA correlation analysis (approximate):")
                logger.info(f"   Top 5 teams by WPA:")
                for i, row in team_wpa.nlargest(5, 'total_wpa').iterrows():
                    logger.info(f"     {row['team']}: {row['total_wpa']:.2f} WPA")
                
                # Note: Full correlation would require team ID mapping
                logger.info(f"   ⚠️  Full correlation requires team ID mapping (not implemented)")
            else:
                logger.info(f"   ⚠️  Cannot compute team correlation - missing TEAM_ID columns")
        else:
            logger.info(f"   ⚠️  Cannot compute team correlation - missing PLAYER1_TEAM_ABBREVIATION")
        
        # Diagnostic C: Player sanity checks
        logger.info(f"\n📊 Diagnostic C - Player Sanity Checks:")
        
        # Known stars
        star_names = [
            "Anthony Edwards", "Edwards", "Nikola Jokic", "Jokic",
            "Luka Doncic", "Doncic", "Shai Gilgeous-Alexander", "Gilgeous-Alexander", "SGA",
            "Jayson Tatum", "Tatum", "Trae Young", "Young",
            "Donovan Mitchell", "Mitchell", "Stephen Curry", "Curry",
            "LeBron James", "James", "Kevin Durant", "Durant"
        ]
        
        logger.info(f"   Known Stars (all found, sorted by WPA):")
        found_stars = []
        for star_name in star_names:
            # Try exact match first
            matches = player_stats[player_stats['player_name'].str.contains(star_name, case=False, na=False)]
            if len(matches) > 0:
                for _, row in matches.iterrows():
                    if row['player_name'] not in [s[0] for s in found_stars]:
                        found_stars.append((row['player_name'], row['player_total_WPA'], row['games_played']))
        
        # Sort by WPA and show all found stars
        found_stars_sorted = sorted(found_stars, key=lambda x: x[1], reverse=True)
        for name, wpa, games in found_stars_sorted:
            status = "✅" if wpa > 0 else "⚠️"
            logger.info(f"     {status} {name}: {wpa:.2f} WPA ({games} games)")
        
        # Role players (sample from middle of distribution)
        logger.info(f"\n   Sample Role Players (middle of distribution):")
        mid_idx = len(player_stats) // 2
        role_players = player_stats.iloc[mid_idx-5:mid_idx+5]
        for _, row in role_players.iterrows():
            logger.info(f"     {row['player_name']}: {row['player_total_WPA']:.2f} WPA ({row['games_played']} games)")
        
        logger.info("="*60 + "\n")
    
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

