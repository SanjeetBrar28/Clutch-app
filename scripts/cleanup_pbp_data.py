#!/usr/bin/env python3
"""
Utility script to clean up and reorganize existing play-by-play data files.

This script helps reorganize the messy flat directory structure into
the new organized team/season structure.
"""

import os
import shutil
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_ingestion import GameUtils


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean up and reorganize existing play-by-play data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.cleanup_pbp_data --action list
  python -m scripts.cleanup_pbp_data --action organize --team "Indiana Pacers" --season "2012-13"
  python -m scripts.cleanup_pbp_data --action clean --confirm
        """
    )
    
    parser.add_argument(
        '--action',
        type=str,
        choices=['list', 'organize', 'clean'],
        required=True,
        help='Action to perform: list files, organize into team/season folders, or clean up'
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw/playbyplay',
        help='Directory containing raw play-by-play files (default: data/raw/playbyplay)'
    )
    parser.add_argument(
        '--team',
        type=str,
        help='Team name for organization (required for organize action)'
    )
    parser.add_argument(
        '--season',
        type=str,
        help='Season for organization (required for organize action)'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm destructive operations (required for clean action)'
    )
    
    return parser.parse_args()


def list_files(raw_dir: str):
    """List all CSV files in the raw directory."""
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist")
        return
    
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    print(f"📁 Found {len(csv_files)} CSV files in {raw_dir}")
    print("\nFiles:")
    for i, file in enumerate(sorted(csv_files), 1):
        print(f"  {i:3d}. {file}")
    
    if len(csv_files) > 0:
        print(f"\nTotal: {len(csv_files)} files")
        
        # Show file sizes
        total_size = 0
        for file in csv_files:
            filepath = os.path.join(raw_dir, file)
            size = os.path.getsize(filepath)
            total_size += size
        
        print(f"Total size: {total_size:,} bytes ({total_size / (1024*1024):.1f} MB)")


def organize_files(raw_dir: str, team_name: str, season: str):
    """Organize files into team/season directory structure."""
    if not team_name or not season:
        print("Error: --team and --season are required for organize action")
        return
    
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        return
    
    # Create organized directory structure
    safe_team_name = team_name.replace(' ', '_').replace('.', '').lower()
    safe_season = season.replace('-', '_')
    team_season_dir = os.path.join(raw_dir, safe_team_name, safe_season)
    
    print(f"📁 Organizing {len(csv_files)} files into: {team_season_dir}")
    
    # Create directory
    os.makedirs(team_season_dir, exist_ok=True)
    
    # Move files
    moved_count = 0
    for file in csv_files:
        old_path = os.path.join(raw_dir, file)
        new_path = os.path.join(team_season_dir, file)
        
        try:
            shutil.move(old_path, new_path)
            moved_count += 1
            print(f"  ✅ Moved: {file}")
        except Exception as e:
            print(f"  ❌ Failed to move {file}: {e}")
    
    print(f"\n🎉 Successfully organized {moved_count}/{len(csv_files)} files")
    print(f"📁 Files now located in: {team_season_dir}")


def clean_files(raw_dir: str, confirm: bool):
    """Clean up files (delete them)."""
    if not confirm:
        print("Error: --confirm flag is required for clean action")
        print("This will permanently delete all CSV files!")
        return
    
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        return
    
    print(f"⚠️  WARNING: This will permanently delete {len(csv_files)} CSV files!")
    print(f"Directory: {raw_dir}")
    
    # Double confirmation
    response = input("\nType 'DELETE' to confirm: ")
    if response != 'DELETE':
        print("Operation cancelled")
        return
    
    deleted_count = 0
    for file in csv_files:
        filepath = os.path.join(raw_dir, file)
        try:
            os.remove(filepath)
            deleted_count += 1
            print(f"  🗑️  Deleted: {file}")
        except Exception as e:
            print(f"  ❌ Failed to delete {file}: {e}")
    
    print(f"\n🎉 Successfully deleted {deleted_count}/{len(csv_files)} files")


def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"Clutch Play-by-Play Data Cleanup - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Action: {args.action}")
    print(f"Directory: {args.raw_dir}")
    print("-" * 60)
    
    try:
        if args.action == 'list':
            list_files(args.raw_dir)
            
        elif args.action == 'organize':
            organize_files(args.raw_dir, args.team, args.season)
            
        elif args.action == 'clean':
            clean_files(args.raw_dir, args.confirm)
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
