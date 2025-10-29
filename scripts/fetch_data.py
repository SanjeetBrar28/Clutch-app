#!/usr/bin/env python3
"""
CLI script for fetching NBA data for Clutch sports analytics.

This script provides a command-line interface to fetch player and team
game log data from the NBA API and save it to CSV files.
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_ingestion import NBADataIngestion, DataIngestionError


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch NBA data for Clutch sports analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_data.py --player "LeBron James" --season "2023-24"
  python scripts/fetch_data.py --team "Lakers" --season "2023-24"
  python scripts/fetch_data.py --player "Stephen Curry" --season "2022-23" --verify
        """
    )
    
    # Data type selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--player',
        type=str,
        help='Full name of the player to fetch data for'
    )
    group.add_argument(
        '--team',
        type=str,
        help='Full name or abbreviation of the team to fetch data for'
    )
    
    # Required arguments
    parser.add_argument(
        '--season',
        type=str,
        required=True,
        help='Season in format "YYYY-YY" (e.g., "2023-24")'
    )
    
    # Optional arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Base directory for data storage (default: data)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.6,
        help='Rate limit delay in seconds (default: 0.6)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify data quality after fetching'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_season_format(season: str) -> bool:
    """
    Validate season format.
    
    Args:
        season: Season string to validate
        
    Returns:
        True if valid format, False otherwise
    """
    try:
        # Check format YYYY-YY
        if len(season) != 7 or season[4] != '-':
            return False
        
        year1 = int(season[:4])
        year2_suffix = int(season[5:])
        
        # Check if second year suffix matches the next year
        expected_year2_suffix = (year1 + 1) % 100
        if year2_suffix != expected_year2_suffix:
            return False
            
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
        print(f"Error: Invalid season format '{args.season}'. Expected format: YYYY-YY")
        sys.exit(1)
    
    # Initialize data ingestion service
    try:
        ingestion = NBADataIngestion(
            data_dir=args.data_dir,
            rate_limit_delay=args.rate_limit
        )
    except Exception as e:
        print(f"Error initializing data ingestion service: {e}")
        sys.exit(1)
    
    print(f"Clutch Data Ingestion - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Season: {args.season}")
    print(f"Data directory: {args.data_dir}")
    print("-" * 50)
    
    try:
        if args.player:
            print(f"Fetching data for player: {args.player}")
            result = ingestion.fetch_and_save_player_data(args.player, args.season)
            
        elif args.team:
            print(f"Fetching data for team: {args.team}")
            result = ingestion.fetch_and_save_team_data(args.team, args.season)
        
        # Display results
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"📁 File saved: {result['filepath']}")
            print(f"📊 Rows: {result['rows']}")
            
            # Verify data quality if requested
            if args.verify and result['filepath']:
                print("\n🔍 Verifying data quality...")
                quality_report = ingestion.verify_data_quality(result['filepath'])
                
                if quality_report['is_readable']:
                    print(f"✅ File is readable")
                    print(f"📊 Columns: {len(quality_report['columns'])}")
                    print(f"💾 Memory usage: {quality_report['memory_usage']:,} bytes")
                    
                    # Show null counts for key columns
                    null_counts = quality_report['has_nulls']
                    if null_counts:
                        print("🔍 Null value counts:")
                        for col, count in null_counts.items():
                            if count > 0:
                                print(f"   {col}: {count}")
                else:
                    print(f"❌ File verification failed: {quality_report['error']}")
            
            # Log success to metadata file
            log_entry = f"{datetime.now().isoformat()} - SUCCESS - {result['message']} - File: {result['filepath']} - Rows: {result['rows']}\n"
            metadata_file = os.path.join(args.data_dir, 'logs', 'fetch_metadata.txt')
            with open(metadata_file, 'a') as f:
                f.write(log_entry)
            
        else:
            print(f"❌ {result['message']}")
            sys.exit(1)
            
    except DataIngestionError as e:
        print(f"❌ Data ingestion error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 Data ingestion completed successfully!")


if __name__ == "__main__":
    main()
