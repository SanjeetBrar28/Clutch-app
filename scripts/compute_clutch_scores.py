"""
End-to-end orchestrator for computing Win Probability (WP) and Win Probability Added (WPA) metrics.

This script:
1. Loads processed play-by-play data
2. Builds features & leverage weights
3. Trains or loads WP model
4. Computes WPA per event
5. Aggregates to player-level Clutch Scores
6. Saves outputs and visualizations
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.wp_model import WinProbabilityModel
from models.clutch_score import ClutchScoreCalculator
from features.build_leverage_features import build_wp_features, compute_leverage_weight
from backend.services.data_ingestion.game_utils import GameUtils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_game_outcomes_from_api(game_ids: list, team_name: str, season: str) -> pd.DataFrame:
    """
    Fetch game outcomes (win/loss) from NBA API using LeagueGameFinder.
    
    This method uses LeagueGameFinder which is more reliable than BoxScoreSummaryV2.
    
    Args:
        game_ids: List of game IDs
        team_name: Team name for determining home/away
        season: Season in format "YYYY-YY"
        
    Returns:
        DataFrame with columns ['GAME_ID', 'home_win'] (1 if home won, 0 if away won)
    """
    logger.info(f"Fetching game outcomes from API for {len(game_ids)} games...")
    
    from nba_api.stats.endpoints import LeagueGameFinder
    
    game_utils = GameUtils()
    team_info = game_utils.get_team_info(team_name)
    
    if team_info is None:
        raise ValueError(f"Team '{team_name}' not found")
    
    team_id = team_info['id']
    
    # Convert season format for API
    if '-' in season:
        nba_season = season
    else:
        year = int(season)
        nba_season = f"{year}-{str(year+1)[2:]}"
    
    try:
        game_utils._rate_limit()
        
        # Use LeagueGameFinder to get game outcomes
        league_games = LeagueGameFinder(
            team_id_nullable=team_id,
            season_nullable=nba_season
        )
        
        games_df = league_games.get_data_frames()[0]
        
        if games_df.empty:
            logger.warning("No games found from API, falling back to inference")
            return pd.DataFrame()
        
        # Create outcomes DataFrame
        game_outcomes = []
        
        for _, game_row in games_df.iterrows():
            game_id = str(game_row['GAME_ID'])
            
            # Only process games in our list
            if game_id not in [str(gid) for gid in game_ids]:
                continue
            
            # Determine winner from WL (Win/Loss) column
            # WL = 'W' if team won, 'L' if team lost
            # We need to check if the team in the row is home or away
            is_home_game = game_row.get('MATCHUP', '').startswith(team_info['abbreviation'])
            
            # If team is home and won, home_win = 1
            # If team is away and won, home_win = 0
            if is_home_game:
                home_win = 1 if game_row.get('WL', '') == 'W' else 0
            else:
                home_win = 1 if game_row.get('WL', '') == 'L' else 0  # If away team lost, home won
            
            game_outcomes.append({
                'GAME_ID': game_id,
                'home_win': home_win
            })
        
        df = pd.DataFrame(game_outcomes)
        logger.info(f"Fetched outcomes for {len(df)} games from API")
        logger.info(f"Home wins: {df['home_win'].sum()}, Away wins: {len(df) - df['home_win'].sum()}")
        
        return df
        
    except Exception as e:
        logger.warning(f"Error fetching game outcomes from API: {str(e)}")
        logger.warning("Falling back to inference from play-by-play data...")
        return pd.DataFrame()


def infer_game_outcomes_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer game outcomes from final event states in play-by-play data.
    
    This method infers winners by looking at the final score margin in the data.
    If score_margin_int > 0 at the end, home team won; if < 0, away team won.
    
    Args:
        df: DataFrame with play-by-play data
        
    Returns:
        DataFrame with columns ['GAME_ID', 'home_win']
    """
    logger.info("Inferring game outcomes from play-by-play data...")
    
    game_outcomes = []
    
    # Get final event for each game (sorted by EVENTNUM)
    for game_id in df['GAME_ID'].unique():
        game_events = df[df['GAME_ID'] == game_id].sort_values('EVENTNUM').reset_index(drop=True)
        
        if len(game_events) == 0:
            continue
        
        # Get final score margin - use the last non-zero margin if final event has margin = 0
        # This handles cases where final events are substitutions/fouls that don't change score
        final_margin = game_events['score_margin_int'].iloc[-1]
        
        # If final margin is 0, look backwards to find the last non-zero margin
        if final_margin == 0:
            # Reverse search for last non-zero margin
            non_zero_margins = game_events[game_events['score_margin_int'] != 0]
            if len(non_zero_margins) > 0:
                final_margin = non_zero_margins['score_margin_int'].iloc[-1]
            else:
                # If all margins are 0 (shouldn't happen), use the max absolute margin
                # This is a fallback for edge cases
                max_abs_margin_idx = game_events['score_margin_int'].abs().idxmax()
                final_margin = game_events.loc[max_abs_margin_idx, 'score_margin_int']
        
        if pd.isna(final_margin):
            # If margin is still missing, default to 0 (away win) - less biased than defaulting to home
            home_win = 0
        elif final_margin > 0:
            home_win = 1  # Home team won
        elif final_margin < 0:
            home_win = 0  # Away team won
        else:
            # Margin is exactly 0 (tied game) - use is_home as tie-breaker
            # If Pacers are home, default to home win; if away, default to away win
            # This is less biased than always defaulting to home
            is_home = game_events['is_home'].iloc[-1] if 'is_home' in game_events.columns else True
            home_win = 1 if is_home else 0
        
        game_outcomes.append({
            'GAME_ID': game_id,
            'home_win': home_win
        })
    
    df_outcomes = pd.DataFrame(game_outcomes)
    logger.info(f"Inferred outcomes for {len(df_outcomes)} games from play-by-play data")
    logger.info(f"Home wins: {df_outcomes['home_win'].sum()}, Away wins: {len(df_outcomes) - df_outcomes['home_win'].sum()}")
    
    return df_outcomes


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute Win Probability Added (WPA) and Clutch Scores for players',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute Clutch Scores for Indiana Pacers 2024-25 season
  python -m scripts.compute_clutch_scores --team "Indiana Pacers" --season "2024-25"
  
  # Retrain WP model
  python -m scripts.compute_clutch_scores --team "Indiana Pacers" --season "2024-25" --retrain
  
  # Use custom data path
  python -m scripts.compute_clutch_scores --team "Indiana Pacers" --season "2024-25" \\
    --data-path data/processed/playbyplay_indiana_pacers_2024_25_cleaned.csv
        """
    )
    
    parser.add_argument(
        '--team',
        type=str,
        required=True,
        help='Team name (e.g., "Indiana Pacers")'
    )
    
    parser.add_argument(
        '--season',
        type=str,
        required=True,
        help='Season in format "YYYY-YY" or "YYYY" (e.g., "2024-25" or "2024")'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to processed play-by-play CSV (default: auto-detect from team/season)'
    )
    
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Retrain WP model even if existing model exists'
    )
    
    parser.add_argument(
        '--use-api-outcomes',
        action='store_true',
        default=False,
        help='Fetch game outcomes from NBA API (default: False, uses inference from data)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/artifacts',
        help='Directory to save/load WP model (default: models/artifacts)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Directory to save outputs (default: data/processed)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        logger.info("="*60)
        logger.info("Clutch Score Computation Pipeline")
        logger.info("="*60)
        logger.info(f"Team: {args.team}")
        logger.info(f"Season: {args.season}")
        logger.info(f"Retrain model: {args.retrain}")
        logger.info("")
        
        # Determine data path
        if args.data_path:
            data_path = args.data_path
        else:
            # Auto-detect from team/season
            safe_team_name = args.team.replace(' ', '_').replace('.', '').lower()
            safe_season = args.season.replace('-', '_')
            data_path = os.path.join(
                'data', 'processed',
                f"playbyplay_{safe_team_name}_{safe_season}_cleaned.csv"
            )
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            logger.error("Please run process_pbp_season_v3.py first to generate processed data")
            sys.exit(1)
        
        logger.info(f"Loading play-by-play data from {data_path}...")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df):,} events from {df['GAME_ID'].nunique()} games")
        
        # Get unique game IDs
        game_ids = df['GAME_ID'].unique().tolist()
        logger.info(f"Processing {len(game_ids)} games")
        
        # Fetch game outcomes
        # Try API first if requested, but fall back to inference if it fails
        game_outcomes = pd.DataFrame()
        
        if args.use_api_outcomes:
            try:
                game_outcomes = fetch_game_outcomes_from_api(game_ids, args.team, args.season)
                if game_outcomes.empty:
                    logger.warning("API returned no outcomes, falling back to inference...")
                    game_outcomes = infer_game_outcomes_from_data(df)
            except Exception as e:
                logger.warning(f"Failed to fetch game outcomes from API: {str(e)}")
                logger.warning("Falling back to inferred outcomes...")
                game_outcomes = infer_game_outcomes_from_data(df)
        else:
            game_outcomes = infer_game_outcomes_from_data(df)
        
        # Ensure all games have outcomes
        missing_games = set(game_ids) - set(game_outcomes['GAME_ID'].unique())
        if missing_games:
            logger.warning(f"Missing outcomes for {len(missing_games)} games, using defaults")
            for game_id in missing_games:
                game_outcomes = pd.concat([
                    game_outcomes,
                    pd.DataFrame([{'GAME_ID': game_id, 'home_win': 1}])
                ], ignore_index=True)
        
        # Build features
        logger.info("\n🔧 Building features...")
        df_features = build_wp_features(df)
        
        # Initialize WP model
        logger.info("\n📊 Initializing Win Probability model...")
        wp_model = WinProbabilityModel(model_dir=args.model_dir, random_state=42)
        
        # Train or load model
        logger.info("\n🎓 Training/loading WP model...")
        train_metrics = wp_model.train(
            df=df_features,
            game_outcomes=game_outcomes,
            retrain=args.retrain
        )
        
        logger.info(f"Model metrics: {train_metrics}")
        
        # Evaluate calibration
        logger.info("\n📈 Evaluating model calibration...")
        calibration = wp_model.evaluate_calibration(df_features, game_outcomes)
        
        # Plot calibration
        calibration_plot_path = os.path.join(args.output_dir, 'wp_calibration_plot.png')
        wp_model.plot_calibration(df_features, game_outcomes, save_path=calibration_plot_path)
        logger.info(f"Calibration plot saved to {calibration_plot_path}")
        
        # Initialize Clutch Score Calculator
        logger.info("\n⚡ Computing Win Probability Added (WPA)...")
        calculator = ClutchScoreCalculator(wp_model)
        
        # Compute event-level WPA
        wpa_df = calculator.compute_event_wpa(df_features, game_outcomes)
        
        # Attribute WPA to players
        wpa_df = calculator.attribute_wpa_to_players(wpa_df)
        
        # Aggregate to player level
        logger.info("\n👥 Aggregating player Clutch Scores...")
        player_stats = calculator.aggregate_player_wpa(wpa_df)
        
        # Save outputs
        logger.info("\n💾 Saving outputs...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save event-level WPA
        wpa_output_path = os.path.join(args.output_dir, 'playbyplay_wpa.csv')
        calculator.save_wpa_data(wpa_df, wpa_output_path)
        
        # Save player summary
        player_output_path = os.path.join(args.output_dir, 'player_wpa_summary.csv')
        calculator.save_player_summary(player_stats, player_output_path)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"\n✅ Processed {len(wpa_df):,} events")
        logger.info(f"✅ Computed WPA for {len(player_stats)} players")
        logger.info(f"\n🏆 Top 5 Players by Total WPA:")
        for i, row in player_stats.head(5).iterrows():
            logger.info(f"   {i+1}. {row['player_name']}: {row['player_total_WPA']:.4f} "
                       f"({row['player_event_count']} events, {row['games_played']} games)")
        
        logger.info(f"\n📉 Bottom 5 Players by Total WPA:")
        for i, row in player_stats.tail(5).iterrows():
            logger.info(f"   {i+1}. {row['player_name']}: {row['player_total_WPA']:.4f} "
                       f"({row['player_event_count']} events, {row['games_played']} games)")
        
        logger.info(f"\n📁 Output files:")
        logger.info(f"   Event-level WPA: {wpa_output_path}")
        logger.info(f"   Player summary: {player_output_path}")
        logger.info(f"   Calibration plot: {calibration_plot_path}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Clutch Score computation completed successfully!")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

