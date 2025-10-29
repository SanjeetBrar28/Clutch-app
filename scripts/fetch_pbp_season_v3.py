#!/usr/bin/env python3
"""
CLI script for fetching play-by-play data for a team's entire season.

This script orchestrates the fetching of play-by-play data for all games
in a team's season using the playbyplayv3 endpoint.

Usage:
    python -m scripts.fetch_pbp_season_v3 --team "Indiana Pacers" --season "2024-25"
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_ingestion import GameUtils, PlayByPlayFetcher


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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch play-by-play data for a team's entire season",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.fetch_pbp_season_v3 --team "Indiana Pacers" --season "2024-25"
  python -m scripts.fetch_pbp_season_v3 --team "Indiana Pacers" --season "2024"
  python -m scripts.fetch_pbp_season_v3 --team "IND" --season "2024-25" --rate-limit 1.0
  python -m scripts.fetch_pbp_season_v3 --team "Indiana Pacers" --season "2024-25" --verbose
        """
    )
    
    parser.add_argument(
        '--team',
        type=str,
        default='Indiana Pacers',
        help='Team name or abbreviation (default: Indiana Pacers)'
    )
    parser.add_argument(
        '--season',
        type=str,
        default='2024-25',
        help='Season in format "YYYY-YY" (e.g., "2024-25") or "YYYY" (e.g., "2024") (default: 2024-25)'
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw/playbyplay',
        help='Directory to save raw play-by-play files (default: data/raw/playbyplay)'
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
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_season_format(season: str) -> bool:
    """Validate season format."""
    try:
        # Accept both "YYYY-YY" and "YYYY" formats
        if '-' in season:
            # Format: YYYY-YY
            if len(season) != 7 or season[4] != '-':
                return False
            
            year1 = int(season[:4])
            year2_suffix = int(season[5:])
            
            expected_year2_suffix = (year1 + 1) % 100
            if year2_suffix != expected_year2_suffix:
                return False
        else:
            # Format: YYYY
            if len(season) != 4:
                return False
            
            year1 = int(season)
        
        # Check reasonable year range (NBA started in 1946)
        if year1 < 1946 or year1 > 2030:
            return False
            
        return True
        
    except ValueError:
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate season format
    if not validate_season_format(args.season):
        print(f"Error: Invalid season format '{args.season}'. Expected format: YYYY-YY (e.g., '2024-25') or YYYY (e.g., '2024')")
        sys.exit(1)
    
    # Setup logging
    log_file = os.path.join('data', 'logs', f'ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    print(f"Clutch Play-by-Play Data Fetcher - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Team: {args.team}")
    print(f"Season: {args.season}")
    print(f"Output Directory: {args.raw_dir}")
    print(f"Rate Limit: {args.rate_limit}s")
    print("-" * 60)
    
    try:
        # Initialize services
        game_utils = GameUtils(rate_limit_delay=args.rate_limit)
        fetcher = PlayByPlayFetcher(
            output_dir=args.raw_dir,
            rate_limit_delay=args.rate_limit
        )
        
        # Get team info and validate
        team_info = game_utils.get_team_info(args.team)
        if team_info is None:
            print(f"Error: Team '{args.team}' not found")
            sys.exit(1)
        
        print(f"✅ Found team: {team_info['full_name']} ({team_info['abbreviation']})")
        
        # Fetch game IDs
        print(f"\n🔍 Fetching game IDs for {args.team} in {args.season}...")
        games = game_utils.fetch_game_ids(args.team, args.season)
        
        if not games:
            print(f"❌ No games found for {args.team} in {args.season}")
            sys.exit(1)
        
        print(f"✅ Found {len(games)} games")
        
        # Get season summary
        summary = game_utils.get_season_summary(args.team, args.season)
        print(f"📊 Season Summary:")
        print(f"   Games: {summary['total_games']}")
        print(f"   Record: {summary['wins']}-{summary['losses']} ({summary['win_percentage']:.1%})")
        print(f"   Avg Points: {summary['avg_points']:.1f}")
        
        # Extract game IDs
        game_ids = [game['GAME_ID'] for game in games]
        
        # Fetch play-by-play data
        print(f"\n🏀 Fetching play-by-play data for {len(game_ids)} games...")
        print("This may take several minutes due to rate limiting...")
        
        results = []
        for game_id in tqdm(game_ids, desc="Fetching games"):
            result = fetcher.fetch_and_save_game(game_id, args.team, args.season)
            results.append(result)
        
        # Get fetch summary
        fetch_summary = fetcher.get_fetch_summary(results)
        
        print(f"\n📈 Fetch Results:")
        print(f"   Successful: {fetch_summary['successful_games']}/{fetch_summary['total_games']}")
        print(f"   Success Rate: {fetch_summary['success_rate']:.1%}")
        print(f"   Total Plays: {fetch_summary['total_plays']:,}")
        print(f"   Avg Plays/Game: {fetch_summary['avg_plays_per_game']:.1f}")
        
        if fetch_summary['failed_games']:
            print(f"   Failed Games: {', '.join(fetch_summary['failed_games'])}")
        
        # Log final results
        logger.info(f"Completed fetching play-by-play data for {args.team} {args.season}")
        logger.info(f"Results: {fetch_summary}")
        
        print(f"\n🎉 Data fetching completed!")
        print(f"📁 Raw files saved to: {args.raw_dir}")
        print(f"📝 Log saved to: {log_file}")
        
        if fetch_summary['success_rate'] < 1.0:
            print(f"\n⚠️  Some games failed to fetch. Check the log for details.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
