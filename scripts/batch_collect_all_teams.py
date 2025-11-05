#!/usr/bin/env python3
"""
Batch orchestration script for collecting and processing play-by-play data
for all NBA teams across multiple seasons.

This script:
1. Fetches play-by-play data for all 30 NBA teams
2. Processes 5 seasons of data (2019-20 through 2024-25)
3. Handles errors gracefully with resume capability
4. Tracks progress and provides detailed reporting

Usage:
    python -m scripts.batch_collect_all_teams --seasons 5
    python -m scripts.batch_collect_all_teams --seasons 3 --teams "Lakers" "Warriors"
    python -m scripts.batch_collect_all_teams --skip-fetch --only-process
"""

import argparse
import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Set
from tqdm import tqdm

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nba_api.stats.static import teams
from backend.services.data_ingestion import GameUtils, PlayByPlayFetcher, PlayByPlayProcessor


def setup_logging(log_file: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_all_seasons(num_seasons: int = 5) -> List[str]:
    """
    Get list of seasons for the last N seasons.
    
    Args:
        num_seasons: Number of seasons to include
        
    Returns:
        List of season strings in "YYYY-YY" format
    """
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # If we're before October, current season hasn't started yet
    if current_month < 10:
        current_year -= 1
    
    seasons = []
    for i in range(num_seasons):
        year = current_year - i
        season_str = f"{year}-{str(year+1)[2:]}"
        seasons.append(season_str)
    
    return seasons


def get_all_teams(filter_teams: List[str] = None) -> List[Dict]:
    """
    Get all NBA teams, optionally filtered.
    
    Args:
        filter_teams: Optional list of team names/abbreviations to include
        
    Returns:
        List of team dictionaries
    """
    all_teams = teams.get_teams()
    
    if filter_teams:
        filtered = []
        filter_lower = [t.lower() for t in filter_teams]
        for team in all_teams:
            if (team['full_name'].lower() in filter_lower or
                team['abbreviation'].lower() in filter_lower):
                filtered.append(team)
        return filtered
    
    return all_teams


def check_data_exists(team_name: str, season: str, raw_dir: str, 
                     processed_dir: str) -> Tuple[bool, bool]:
    """
    Check if raw and processed data already exist for a team/season.
    
    Returns:
        Tuple of (raw_exists, processed_exists)
    """
    safe_team_name = team_name.replace(' ', '_').replace('.', '').lower()
    safe_season = season.replace('-', '_')
    
    # Check raw data
    raw_team_season_dir = os.path.join(raw_dir, safe_team_name, safe_season)
    raw_exists = os.path.exists(raw_team_season_dir) and len([f for f in os.listdir(raw_team_season_dir) if f.endswith('.csv')]) > 0
    
    # Check processed data
    processed_filename = f"playbyplay_{safe_team_name}_{safe_season}_cleaned.csv"
    processed_path = os.path.join(processed_dir, processed_filename)
    processed_exists = os.path.exists(processed_path)
    
    return raw_exists, processed_exists


def fetch_team_season(team: Dict, season: str, raw_dir: str, 
                     rate_limit: float, max_retries: int, 
                     skip_existing: bool = True) -> Dict:
    """
    Fetch play-by-play data for a team/season.
    
    Returns:
        Dictionary with fetch results
    """
    team_name = team['full_name']
    team_abbrev = team['abbreviation']
    
    result = {
        'team': team_name,
        'season': season,
        'success': False,
        'games_fetched': 0,
        'games_failed': 0,
        'total_plays': 0,
        'error': None,
        'skipped': False
    }
    
    try:
        # Check if already exists
        raw_exists, _ = check_data_exists(team_name, season, raw_dir, '')
        if skip_existing and raw_exists:
            result['skipped'] = True
            result['success'] = True
            # Count existing games
            safe_team_name = team_name.replace(' ', '_').replace('.', '').lower()
            safe_season = season.replace('-', '_')
            raw_dir_path = os.path.join(raw_dir, safe_team_name, safe_season)
            result['games_fetched'] = len([f for f in os.listdir(raw_dir_path) if f.endswith('.csv')])
            return result
        
        # Initialize services
        game_utils = GameUtils(rate_limit_delay=rate_limit)
        fetcher = PlayByPlayFetcher(output_dir=raw_dir, rate_limit_delay=rate_limit)
        
        # Fetch game IDs
        games = game_utils.fetch_game_ids(team_name, season, max_retries=max_retries)
        
        if not games:
            result['error'] = "No games found"
            return result
        
        # Fetch play-by-play data
        game_ids = [game['GAME_ID'] for game in games]
        fetch_results = []
        
        for game_id in game_ids:
            fetch_result = fetcher.fetch_and_save_game(game_id, team_name, season)
            fetch_results.append(fetch_result)
            if fetch_result['success']:
                result['games_fetched'] += 1
                result['total_plays'] += fetch_result.get('num_events', 0)
            else:
                result['games_failed'] += 1
        
        result['success'] = result['games_fetched'] > 0
        
    except Exception as e:
        result['error'] = str(e)
        logging.error(f"Error fetching {team_name} {season}: {e}")
    
    return result


def process_team_season(team: Dict, season: str, raw_dir: str, 
                       processed_dir: str, skip_existing: bool = True) -> Dict:
    """
    Process play-by-play data for a team/season.
    
    Returns:
        Dictionary with processing results
    """
    team_name = team['full_name']
    team_abbrev = team['abbreviation']
    
    result = {
        'team': team_name,
        'season': season,
        'success': False,
        'rows_processed': 0,
        'games_processed': 0,
        'error': None,
        'skipped': False
    }
    
    try:
        # Check if already exists
        _, processed_exists = check_data_exists(team_name, season, raw_dir, processed_dir)
        if skip_existing and processed_exists:
            result['skipped'] = True
            result['success'] = True
            # Load to get stats
            safe_team_name = team_name.replace(' ', '_').replace('.', '').lower()
            safe_season = season.replace('-', '_')
            processed_filename = f"playbyplay_{safe_team_name}_{safe_season}_cleaned.csv"
            processed_path = os.path.join(processed_dir, processed_filename)
            import pandas as pd
            df = pd.read_csv(processed_path)
            result['rows_processed'] = len(df)
            result['games_processed'] = df['GAME_ID'].nunique()
            return result
        
        # Check if raw data exists
        raw_exists, _ = check_data_exists(team_name, season, raw_dir, '')
        if not raw_exists:
            result['error'] = "Raw data not found - must fetch first"
            return result
        
        # Initialize processor
        processor = PlayByPlayProcessor(raw_dir=raw_dir, processed_dir=processed_dir)
        
        # Load raw data
        safe_team_name = team_name.replace(' ', '_').replace('.', '').lower()
        safe_season = season.replace('-', '_')
        raw_team_season_dir = os.path.join(raw_dir, safe_team_name, safe_season)
        csv_files = [f for f in os.listdir(raw_team_season_dir) if f.endswith('.csv')]
        game_ids = [f.replace('.csv', '') for f in csv_files]
        
        games_data = processor.load_all_games(game_ids, team_name, season)
        
        if not games_data:
            result['error'] = "No game data could be loaded"
            return result
        
        # Merge and process
        merged_df = processor.merge_all_games(games_data, team_abbrev)
        
        if merged_df.empty:
            result['error'] = "No data after processing"
            return result
        
        # Validate
        validation_results = processor.validate_data(merged_df)
        
        # Save
        output_filename = f"playbyplay_{safe_team_name}_{safe_season}_cleaned.csv"
        output_path = processor.save_processed_data(merged_df, output_filename)
        
        if output_path:
            result['success'] = True
            result['rows_processed'] = len(merged_df)
            result['games_processed'] = merged_df['GAME_ID'].nunique()
        
    except Exception as e:
        result['error'] = str(e)
        logging.error(f"Error processing {team_name} {season}: {e}")
    
    return result


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch collect and process play-by-play data for all NBA teams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect last 5 seasons for all teams
  python -m scripts.batch_collect_all_teams --seasons 5
  
  # Collect only last 3 seasons for specific teams
  python -m scripts.batch_collect_all_teams --seasons 3 --teams "Lakers" "Warriors"
  
  # Only process existing raw data (skip fetching)
  python -m scripts.batch_collect_all_teams --skip-fetch --only-process
  
  # Re-fetch everything (don't skip existing)
  python -m scripts.batch_collect_all_teams --seasons 5 --force-refetch
        """
    )
    
    parser.add_argument(
        '--seasons',
        type=int,
        default=5,
        help='Number of seasons to collect (default: 5)'
    )
    parser.add_argument(
        '--teams',
        nargs='+',
        help='Specific teams to process (by name or abbreviation). If not specified, processes all teams.'
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw/playbyplay',
        help='Directory for raw play-by-play files (default: data/raw/playbyplay)'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Directory for processed files (default: data/processed)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.6,
        help='Rate limit delay in seconds (default: 0.6)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retry attempts per game (default: 3)'
    )
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip fetching step (only process existing raw data)'
    )
    parser.add_argument(
        '--only-process',
        action='store_true',
        help='Only process data (skip fetching entirely)'
    )
    parser.add_argument(
        '--force-refetch',
        action='store_true',
        help='Re-fetch data even if it already exists'
    )
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Re-process data even if it already exists'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint file for resuming (default: data/logs/batch_checkpoint.json)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    log_file = os.path.join('data', 'logs', f'batch_collect_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    checkpoint_file = args.checkpoint or os.path.join('data', 'logs', 'batch_checkpoint.json')
    
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("CLUTCH BATCH DATA COLLECTION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seasons: {args.seasons}")
    print(f"Skip fetch: {args.skip_fetch or args.only_process}")
    print(f"Force refetch: {args.force_refetch}")
    print(f"Force reprocess: {args.force_reprocess}")
    print("=" * 80)
    
    # Get seasons and teams
    seasons = get_all_seasons(args.seasons)
    teams_list = get_all_teams(args.teams)
    
    print(f"\n📋 Plan:")
    print(f"   Teams: {len(teams_list)}")
    print(f"   Seasons: {len(seasons)}")
    print(f"   Total combinations: {len(teams_list) * len(seasons)}")
    
    # Estimate time
    if not args.skip_fetch and not args.only_process:
        # Rough estimate: 100 games per team/season, 0.6s per game = 60s per team/season
        estimated_seconds = len(teams_list) * len(seasons) * 60
        estimated_hours = estimated_seconds / 3600
        print(f"   ⏱️  Estimated time: ~{estimated_hours:.1f} hours")
        print(f"   ⚠️  This will take a long time! Consider running overnight.")
    
    # Create directories
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    
    # Collect results
    fetch_results = []
    process_results = []
    
    # Main loop
    total_combinations = len(teams_list) * len(seasons)
    
    with tqdm(total=total_combinations * (2 if not args.only_process else 1), 
              desc="Processing") as pbar:
        
        for team in teams_list:
            team_name = team['full_name']
            
            for season in seasons:
                pbar.set_description(f"{team_name[:20]:20s} {season}")
                
                # Fetch step
                if not args.skip_fetch and not args.only_process:
                    fetch_result = fetch_team_season(
                        team, season, args.raw_dir, 
                        args.rate_limit, args.max_retries,
                        skip_existing=not args.force_refetch
                    )
                    fetch_results.append(fetch_result)
                    
                    if fetch_result['skipped']:
                        pbar.set_postfix({"fetch": "skipped", "process": "..."})
                    elif fetch_result['success']:
                        pbar.set_postfix({"fetch": "✓", "process": "..."})
                    else:
                        pbar.set_postfix({"fetch": "✗", "process": "..."})
                
                pbar.update(1)
                
                # Process step
                process_result = process_team_season(
                    team, season, args.raw_dir, args.processed_dir,
                    skip_existing=not args.force_reprocess
                )
                process_results.append(process_result)
                
                if process_result['skipped']:
                    pbar.set_postfix({"fetch": "...", "process": "skipped"})
                elif process_result['success']:
                    pbar.set_postfix({"fetch": "...", "process": "✓"})
                else:
                    pbar.set_postfix({"fetch": "...", "process": "✗"})
                
                if not args.skip_fetch and not args.only_process:
                    pbar.update(1)
                else:
                    pbar.update(1)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if not args.skip_fetch and not args.only_process:
        successful_fetches = sum(1 for r in fetch_results if r['success'] and not r['skipped'])
        skipped_fetches = sum(1 for r in fetch_results if r['skipped'])
        failed_fetches = sum(1 for r in fetch_results if not r['success'] and not r['skipped'])
        total_games_fetched = sum(r['games_fetched'] for r in fetch_results)
        
        print(f"\n📥 Fetch Results:")
        print(f"   Successful: {successful_fetches}")
        print(f"   Skipped: {skipped_fetches}")
        print(f"   Failed: {failed_fetches}")
        print(f"   Total games fetched: {total_games_fetched:,}")
    
    successful_processes = sum(1 for r in process_results if r['success'] and not r['skipped'])
    skipped_processes = sum(1 for r in process_results if r['skipped'])
    failed_processes = sum(1 for r in process_results if not r['success'] and not r['skipped'])
    total_rows_processed = sum(r['rows_processed'] for r in process_results)
    total_games_processed = sum(r['games_processed'] for r in process_results)
    
    print(f"\n📊 Process Results:")
    print(f"   Successful: {successful_processes}")
    print(f"   Skipped: {skipped_processes}")
    print(f"   Failed: {failed_processes}")
    print(f"   Total rows processed: {total_rows_processed:,}")
    print(f"   Total games processed: {total_games_processed:,}")
    
    # Print failures
    if not args.skip_fetch and not args.only_process:
        failed_fetch_details = [r for r in fetch_results if not r['success'] and not r['skipped']]
        if failed_fetch_details:
            print(f"\n❌ Failed Fetches ({len(failed_fetch_details)}):")
            for r in failed_fetch_details[:10]:  # Show first 10
                print(f"   {r['team']} {r['season']}: {r.get('error', 'Unknown error')}")
            if len(failed_fetch_details) > 10:
                print(f"   ... and {len(failed_fetch_details) - 10} more")
    
    failed_process_details = [r for r in process_results if not r['success'] and not r['skipped']]
    if failed_process_details:
        print(f"\n❌ Failed Processes ({len(failed_process_details)}):")
        for r in failed_process_details[:10]:  # Show first 10
            print(f"   {r['team']} {r['season']}: {r.get('error', 'Unknown error')}")
        if len(failed_process_details) > 10:
            print(f"   ... and {len(failed_process_details) - 10} more")
    
    print(f"\n📝 Log saved to: {log_file}")
    print("=" * 80)
    
    logger.info(f"Batch collection completed")
    logger.info(f"Fetch results: {len(fetch_results) if not args.skip_fetch and not args.only_process else 0} total")
    logger.info(f"Process results: {len(process_results)} total")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        print("You can resume later by running the same command (it will skip existing data)")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
