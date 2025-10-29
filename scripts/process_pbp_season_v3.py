#!/usr/bin/env python3
"""
CLI script for processing and merging play-by-play data.

This script loads all raw play-by-play CSV files, cleans and standardizes
the data, and merges everything into a single processed dataset.

Usage:
    python -m scripts.process_pbp_season_v3 --team "Indiana Pacers" --season "2024-25"
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

from backend.services.data_ingestion import GameUtils, PlayByPlayProcessor


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
        description="Process and merge play-by-play data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.process_pbp_season_v3 --team "Indiana Pacers" --season "2024-25"
  python -m scripts.process_pbp_season_v3 --team "Indiana Pacers" --season "2024"
  python -m scripts.process_pbp_season_v3 --team "IND" --season "2024-25" --output "pacers_2024_25.csv"
  python -m scripts.process_pbp_season_v3 --team "Indiana Pacers" --season "2024-25" --verbose
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
        help='Directory containing raw play-by-play files (default: data/raw/playbyplay)'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Directory to save processed data (default: data/processed)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output filename (default: playbyplay_{team}_{season}.csv)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    log_file = os.path.join('data', 'logs', f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    print(f"Clutch Play-by-Play Data Processor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Team: {args.team}")
    print(f"Season: {args.season}")
    print(f"Raw Directory: {args.raw_dir}")
    print(f"Processed Directory: {args.processed_dir}")
    print("-" * 60)
    
    try:
        # Initialize services
        game_utils = GameUtils()
        processor = PlayByPlayProcessor(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir
        )
        
        # Get team info
        team_info = game_utils.get_team_info(args.team)
        if team_info is None:
            print(f"Error: Team '{args.team}' not found")
            sys.exit(1)
        
        team_abbrev = team_info['abbreviation']
        print(f"✅ Processing data for: {team_info['full_name']} ({team_abbrev})")
        
        # Get list of available game files
        if not os.path.exists(args.raw_dir):
            print(f"Error: Raw directory '{args.raw_dir}' does not exist")
            print("Run fetch_pbp_season_v3.py first to fetch raw data")
            sys.exit(1)
        
        # Create organized directory path
        safe_team_name = args.team.replace(' ', '_').replace('.', '').lower()
        safe_season = args.season.replace('-', '_')
        team_season_dir = os.path.join(args.raw_dir, safe_team_name, safe_season)
        
        if not os.path.exists(team_season_dir):
            print(f"Error: Team/season directory '{team_season_dir}' does not exist")
            print("Run fetch_pbp_season_v3.py first to fetch raw data")
            sys.exit(1)
        
        # Find all CSV files in the team/season directory
        csv_files = [f for f in os.listdir(team_season_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"Error: No CSV files found in '{team_season_dir}'")
            print("Run fetch_pbp_season_v3.py first to fetch raw data")
            sys.exit(1)
        
        # Extract game IDs from filenames
        game_ids = [f.replace('.csv', '') for f in csv_files]
        
        print(f"✅ Found {len(game_ids)} game files to process")
        
        # Load all game data
        print(f"\n📂 Loading play-by-play data...")
        games_data = processor.load_all_games(game_ids, args.team, args.season)
        
        if not games_data:
            print("❌ No game data could be loaded")
            sys.exit(1)
        
        print(f"✅ Successfully loaded {len(games_data)} games")
        
        # Merge and clean data
        print(f"\n🔧 Processing and merging data...")
        merged_df = processor.merge_all_games(games_data, team_abbrev)
        
        if merged_df.empty:
            print("❌ No data after processing")
            sys.exit(1)
        
        # Generate output filename
        if args.output:
            output_filename = args.output
        else:
            safe_team_name = team_info['full_name'].replace(' ', '_').lower()
            safe_season = args.season.replace('-', '_')
            output_filename = f"playbyplay_{safe_team_name}_{safe_season}.csv"
        
        # Save processed data
        print(f"\n💾 Saving processed data...")
        output_path = processor.save_processed_data(merged_df, output_filename)
        
        if not output_path:
            print("❌ Failed to save processed data")
            sys.exit(1)
        
        # Get data summary
        summary = processor.get_data_summary(merged_df)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total Plays: {summary['total_plays']:,}")
        print(f"   Total Games: {summary['total_games']}")
        print(f"   Avg Plays/Game: {summary['avg_plays_per_game']:.1f}")
        
        print(f"\n📈 Event Categories:")
        for category, count in summary['event_categories'].items():
            percentage = (count / summary['total_plays']) * 100
            print(f"   {category}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n⏱️  Periods:")
        for period, count in summary['periods'].items():
            print(f"   Period {period}: {count:,} plays")
        
        print(f"\n📊 Score Margin Stats:")
        print(f"   Mean: {summary['score_margin_stats']['mean']:.1f}")
        print(f"   Std: {summary['score_margin_stats']['std']:.1f}")
        print(f"   Range: {summary['score_margin_stats']['min']:.1f} to {summary['score_margin_stats']['max']:.1f}")
        
        # Log final results
        logger.info(f"Completed processing play-by-play data for {args.team} {args.season}")
        logger.info(f"Results: {summary}")
        
        print(f"\n🎉 Data processing completed!")
        print(f"📁 Processed file saved to: {output_path}")
        print(f"📝 Log saved to: {log_file}")
        
        # File size info
        file_size = os.path.getsize(output_path)
        print(f"📏 File size: {file_size:,} bytes ({file_size / (1024*1024):.1f} MB)")
        
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
