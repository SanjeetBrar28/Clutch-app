#!/usr/bin/env python3
"""
Simple test script to verify NBA API connectivity without pandas dependency issues.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_nba_api():
    """Test NBA API connectivity."""
    try:
        from nba_api.stats.static import players, teams
        from nba_api.stats.endpoints import PlayerGameLog
        
        print("Testing NBA API connectivity...")
        
        # Test player lookup
        player_list = players.get_players()
        lebron = next((p for p in player_list if 'LeBron James' in p['full_name']), None)
        
        if lebron:
            print(f"✅ Found LeBron James: ID {lebron['id']}")
        else:
            print("❌ Could not find LeBron James")
            return False
        
        # Test team lookup
        team_list = teams.get_teams()
        lakers = next((t for t in team_list if 'Lakers' in t['full_name']), None)
        
        if lakers:
            print(f"✅ Found Lakers: ID {lakers['id']}")
        else:
            print("❌ Could not find Lakers")
            return False
        
        print("✅ NBA API connectivity test passed!")
        return True
        
    except Exception as e:
        print(f"❌ NBA API test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_nba_api()
    sys.exit(0 if success else 1)
