"""
Validate WPA scores for quality and reasonableness.

This script checks:
1. Do top players make basketball sense?
2. Do WPA values correlate with traditional stats?
3. Are there suspicious outliers?
4. Do Pacers players rank appropriately?
5. Is leverage weighting working?
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_player_rankings(df: pd.DataFrame) -> dict:
    """
    Validate that top players make basketball sense.
    
    Args:
        df: DataFrame with player WPA summary
        
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*70)
    print("1. VALIDATING PLAYER RANKINGS (Basketball Logic)")
    print("="*70)
    
    # Known good players (Pacers starters/known stars)
    known_good_players = ['Haliburton', 'Siakam', 'Mathurin', 'Toppin', 'McConnell']
    
    # Known opponent stars (should have mixed results)
    opponent_stars = ['Gilgeous-Alexander', 'Curry', 'Irving', 'Antetokounmpo', 
                     'Tatum', 'Embiid', 'Dončić', 'Lillard']
    
    validation = {
        'pacers_rank_high': False,
        'top_players_known': [],
        'opponent_stars_reasonable': [],
        'issues': []
    }
    
    top_10 = df.nlargest(10, 'player_total_WPA')
    top_10_names = top_10['player_name'].tolist()
    
    print(f"\n✓ Top 10 Players by WPA:")
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        is_pacer = row['player_name'] in known_good_players
        marker = "⭐ PACER" if is_pacer else ""
        print(f"   {i:2d}. {row['player_name']:20s} {row['player_total_WPA']:7.3f} WPA ({row['player_event_count']:4d} events) {marker}")
        if is_pacer:
            validation['top_players_known'].append(row['player_name'])
    
    # Check if Pacers players rank high
    pacers_in_top_10 = sum(1 for name in top_10_names if name in known_good_players)
    if pacers_in_top_10 >= 3:
        print(f"\n✅ PASS: {pacers_in_top_10}/5 known Pacers starters in top 10")
        validation['pacers_rank_high'] = True
    else:
        print(f"\n⚠️  WARNING: Only {pacers_in_top_10}/5 known Pacers starters in top 10")
        validation['issues'].append(f"Only {pacers_in_top_10} Pacers in top 10")
    
    # Check opponent stars
    print(f"\n✓ Opponent Stars (should have mixed results):")
    for star in opponent_stars:
        star_df = df[df['player_name'] == star]
        if len(star_df) > 0:
            row = star_df.iloc[0]
            rank = df[df['player_total_WPA'] > row['player_total_WPA']].shape[0] + 1
            wpa = row['player_total_WPA']
            marker = "✓" if -1.0 < wpa < 1.0 else "⚠️"
            print(f"   {star:25s} Rank: {rank:3d}, WPA: {wpa:7.3f} {marker}")
            validation['opponent_stars_reasonable'].append({
                'name': star,
                'rank': rank,
                'wpa': wpa,
                'reasonable': -1.0 < wpa < 1.0
            })
    
    return validation


def validate_statistical_properties(df: pd.DataFrame) -> dict:
    """
    Validate statistical properties of WPA scores.
    
    Args:
        df: DataFrame with player WPA summary
        
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*70)
    print("2. VALIDATING STATISTICAL PROPERTIES")
    print("="*70)
    
    validation = {
        'distribution_balanced': False,
        'outliers_detected': [],
        'correlation_with_events': 0.0,
        'issues': []
    }
    
    # Check distribution
    positive_pct = (df['player_total_WPA'] > 0).sum() / len(df) * 100
    negative_pct = (df['player_total_WPA'] < 0).sum() / len(df) * 100
    
    print(f"\n✓ Distribution:")
    print(f"   Positive WPA: {positive_pct:.1f}% of players")
    print(f"   Negative WPA: {negative_pct:.1f}% of players")
    
    if 40 <= positive_pct <= 60:
        print(f"   ✅ PASS: Distribution is balanced (40-60% range)")
        validation['distribution_balanced'] = True
    else:
        print(f"   ⚠️  WARNING: Distribution is skewed (should be ~50/50)")
        validation['issues'].append(f"Unbalanced distribution: {positive_pct:.1f}% positive")
    
    # Check for outliers
    q1 = df['player_total_WPA'].quantile(0.25)
    q3 = df['player_total_WPA'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df['player_total_WPA'] < lower_bound) | (df['player_total_WPA'] > upper_bound)]
    
    print(f"\n✓ Outliers (IQR method):")
    print(f"   Lower bound: {lower_bound:.2f}")
    print(f"   Upper bound: {upper_bound:.2f}")
    print(f"   Outliers found: {len(outliers)} players")
    
    if len(outliers) > 0:
        print(f"   Top 5 outliers:")
        for idx, row in outliers.nlargest(5, 'player_total_WPA').iterrows():
            print(f"      {row['player_name']:20s} {row['player_total_WPA']:7.3f} WPA")
            validation['outliers_detected'].append({
                'name': row['player_name'],
                'wpa': row['player_total_WPA'],
                'events': row['player_event_count']
            })
    
    # Check correlation with event count
    correlation = df['player_total_WPA'].corr(df['player_event_count'])
    validation['correlation_with_events'] = correlation
    
    print(f"\n✓ Correlation with Event Count:")
    print(f"   Correlation: {correlation:.3f}")
    if abs(correlation) < 0.3:
        print(f"   ✅ PASS: Low correlation (WPA is not just event count)")
    else:
        print(f"   ⚠️  WARNING: High correlation (WPA might just be event count)")
        validation['issues'].append(f"High correlation with events: {correlation:.3f}")
    
    # Check mean WPA (should be near 0)
    mean_wpa = df['player_total_WPA'].mean()
    print(f"\n✓ Mean WPA: {mean_wpa:.4f}")
    if abs(mean_wpa) < 0.1:
        print(f"   ✅ PASS: Mean near 0 (balanced)")
    else:
        print(f"   ⚠️  WARNING: Mean far from 0 (might indicate bias)")
        validation['issues'].append(f"Mean WPA far from 0: {mean_wpa:.4f}")
    
    return validation


def validate_leverage_weighting(df: pd.DataFrame, wpa_df: pd.DataFrame) -> dict:
    """
    Validate that leverage weighting is working.
    
    Args:
        df: Player WPA summary
        wpa_df: Event-level WPA data
        
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*70)
    print("3. VALIDATING LEVERAGE WEIGHTING")
    print("="*70)
    
    validation = {
        'leverage_working': False,
        'late_game_emphasis': False,
        'close_game_emphasis': False
    }
    
    # Check if high leverage events have higher WPA_weighted
    late_game = wpa_df[wpa_df['total_seconds_remaining'] < 300]  # Last 5 minutes
    early_game = wpa_df[wpa_df['total_seconds_remaining'] > 2000]  # First ~13 minutes
    
    close_game = wpa_df[abs(wpa_df['score_margin_int']) <= 3]  # Within 3 points
    blowout = wpa_df[abs(wpa_df['score_margin_int']) > 15]  # Blowout
    
    print(f"\n✓ Late Game (last 5 min) vs Early Game:")
    print(f"   Late game mean WPA_weighted: {late_game['WPA_weighted'].abs().mean():.6f}")
    print(f"   Early game mean WPA_weighted: {early_game['WPA_weighted'].abs().mean():.6f}")
    
    if late_game['WPA_weighted'].abs().mean() > early_game['WPA_weighted'].abs().mean():
        print(f"   ✅ PASS: Late game events have higher weighted WPA")
        validation['late_game_emphasis'] = True
    else:
        print(f"   ⚠️  WARNING: Late game events not emphasized enough")
    
    print(f"\n✓ Close Games (≤3 pts) vs Blowouts (>15 pts):")
    print(f"   Close game mean WPA_weighted: {close_game['WPA_weighted'].abs().mean():.6f}")
    print(f"   Blowout mean WPA_weighted: {blowout['WPA_weighted'].abs().mean():.6f}")
    
    if close_game['WPA_weighted'].abs().mean() > blowout['WPA_weighted'].abs().mean():
        print(f"   ✅ PASS: Close games have higher weighted WPA")
        validation['close_game_emphasis'] = True
    else:
        print(f"   ⚠️  WARNING: Close games not emphasized enough")
    
    if validation['late_game_emphasis'] and validation['close_game_emphasis']:
        validation['leverage_working'] = True
    
    # Check leverage index distribution
    print(f"\n✓ Leverage Index Distribution:")
    print(f"   Mean leverage: {df['player_leverage_index'].mean():.3f}")
    print(f"   Min leverage: {df['player_leverage_index'].min():.3f}")
    print(f"   Max leverage: {df['player_leverage_index'].max():.3f}")
    print(f"   Range: 1.0 (baseline) to 3.0 (max)")
    
    return validation


def validate_sample_size(df: pd.DataFrame) -> dict:
    """
    Validate that players have sufficient sample sizes.
    
    Args:
        df: Player WPA summary
        
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*70)
    print("4. VALIDATING SAMPLE SIZES")
    print("="*70)
    
    validation = {
        'top_players_have_samples': False,
        'low_sample_players': []
    }
    
    # Check top players have sufficient events
    top_10 = df.nlargest(10, 'player_total_WPA')
    
    print(f"\n✓ Top 10 Players Sample Sizes:")
    low_sample_count = 0
    for idx, row in top_10.iterrows():
        events = row['player_event_count']
        games = row['games_played']
        marker = "✅" if events >= 100 else "⚠️"
        if events < 100:
            low_sample_count += 1
            validation['low_sample_players'].append({
                'name': row['player_name'],
                'events': events,
                'games': games
            })
        print(f"   {row['player_name']:20s} {events:4d} events, {games:3d} games {marker}")
    
    if low_sample_count == 0:
        print(f"\n✅ PASS: All top 10 players have ≥100 events")
        validation['top_players_have_samples'] = True
    else:
        print(f"\n⚠️  WARNING: {low_sample_count} top players have <100 events")
    
    # Check overall distribution
    high_sample = (df['player_event_count'] >= 500).sum()
    medium_sample = ((df['player_event_count'] >= 100) & (df['player_event_count'] < 500)).sum()
    low_sample = (df['player_event_count'] < 100).sum()
    
    print(f"\n✓ Sample Size Distribution:")
    print(f"   High sample (≥500 events): {high_sample} players")
    print(f"   Medium sample (100-499 events): {medium_sample} players")
    print(f"   Low sample (<100 events): {low_sample} players")
    
    return validation


def generate_validation_report(all_validations: dict):
    """
    Generate overall validation report.
    
    Args:
        all_validations: Dictionary with all validation results
    """
    print("\n" + "="*70)
    print("OVERALL VALIDATION REPORT")
    print("="*70)
    
    total_checks = 0
    passed_checks = 0
    
    # Check rankings
    if all_validations['rankings']['pacers_rank_high']:
        print("✅ Pacers players rank appropriately")
        passed_checks += 1
    else:
        print("❌ Pacers players don't rank high enough")
    total_checks += 1
    
    # Check distribution
    if all_validations['stats']['distribution_balanced']:
        print("✅ WPA distribution is balanced")
        passed_checks += 1
    else:
        print("❌ WPA distribution is skewed")
    total_checks += 1
    
    # Check leverage
    if all_validations['leverage']['leverage_working']:
        print("✅ Leverage weighting is working")
        passed_checks += 1
    else:
        print("❌ Leverage weighting not working properly")
    total_checks += 1
    
    # Check sample sizes
    if all_validations['samples']['top_players_have_samples']:
        print("✅ Top players have sufficient sample sizes")
        passed_checks += 1
    else:
        print("⚠️  Some top players have low sample sizes")
    total_checks += 1
    
    # Check correlation
    if abs(all_validations['stats']['correlation_with_events']) < 0.3:
        print("✅ WPA is independent of event count")
        passed_checks += 1
    else:
        print("⚠️  WPA highly correlated with event count")
    total_checks += 1
    
    print(f"\n{'='*70}")
    print(f"PASSED: {passed_checks}/{total_checks} checks")
    print(f"{'='*70}")
    
    # List issues
    all_issues = []
    all_issues.extend(all_validations['rankings'].get('issues', []))
    all_issues.extend(all_validations['stats'].get('issues', []))
    all_issues.extend(all_validations['leverage'].get('issues', []))
    all_issues.extend(all_validations['samples'].get('issues', []))
    
    if all_issues:
        print(f"\n⚠️  ISSUES FOUND:")
        for issue in all_issues:
            print(f"   • {issue}")
    else:
        print("\n✅ No major issues found!")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS:")
    print("="*70)
    
    if passed_checks >= 4:
        print("✅ WPA scores appear to be reasonable and meaningful")
        print("   You can trust the player rankings for analysis")
    else:
        print("⚠️  WPA scores have some issues")
        print("   Consider investigating the problems above")
        print("   Player rankings may still be useful, but interpret with caution")


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate WPA scores')
    parser.add_argument(
        '--player-summary',
        type=str,
        default='data/processed/player_wpa_summary.csv',
        help='Path to player WPA summary CSV'
    )
    parser.add_argument(
        '--wpa-data',
        type=str,
        default='data/processed/playbyplay_wpa.csv',
        help='Path to event-level WPA CSV'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("WPA SCORE VALIDATION")
    print("="*70)
    print(f"\nLoading data from:")
    print(f"   Player summary: {args.player_summary}")
    print(f"   Event-level WPA: {args.wpa_data}")
    
    # Load data
    player_df = pd.read_csv(args.player_summary)
    wpa_df = pd.read_csv(args.wpa_data)
    
    print(f"\nLoaded {len(player_df)} players and {len(wpa_df):,} events")
    
    # Run validations
    validations = {
        'rankings': validate_player_rankings(player_df),
        'stats': validate_statistical_properties(player_df),
        'leverage': validate_leverage_weighting(player_df, wpa_df),
        'samples': validate_sample_size(player_df)
    }
    
    # Generate report
    generate_validation_report(validations)


if __name__ == "__main__":
    main()

