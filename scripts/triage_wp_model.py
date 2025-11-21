"""
Diagnostic script to debug Win Probability model performance issues.

This script performs comprehensive integrity checks, visual diagnostics,
and model evaluation to identify why AUC might be around 0.5.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    brier_score_loss, log_loss, roc_curve
)
from sklearn.calibration import calibration_curve

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compute_clutch_scores import infer_game_outcomes_from_data
from features.build_leverage_features import build_wp_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'data/logs/triage_wp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load play-by-play data from CSV."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def get_game_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Infer game outcomes from play-by-play data."""
    logger.info("Inferring game outcomes from play-by-play data...")
    game_outcomes = infer_game_outcomes_from_data(df)
    return game_outcomes


def convert_to_home_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert score_margin_int to score_margin_home (from home team's perspective).
    
    The score_margin_int is from the tracked team's perspective.
    We need to flip the sign if the team is away to get home perspective.
    """
    df = df.copy()
    
    # Convert is_home to boolean/int if needed
    if df['is_home'].dtype == 'object':
        df['is_home'] = df['is_home'].astype(bool).astype(int)
    else:
        df['is_home'] = df['is_home'].astype(int)
    
    # score_margin_int is from tracked team's perspective
    # If is_home = 1, margin is already from home perspective
    # If is_home = 0, flip the sign to get home perspective
    df['score_margin_home'] = np.where(
        df['is_home'] == 1,
        df['score_margin_int'],
        -df['score_margin_int']
    )
    
    return df


def perform_integrity_checks(df: pd.DataFrame, game_outcomes: pd.DataFrame) -> Dict:
    """Perform core integrity checks on the dataset."""
    logger.info("=" * 80)
    logger.info("PERFORMING INTEGRITY CHECKS")
    logger.info("=" * 80)
    
    checks = {
        'total_rows': len(df),
        'unique_games': df['GAME_ID'].nunique(),
        'features_available': list(df.columns),
        'label_balance': {},
        'label_consistency': True,
        'margin_sign_symmetry': True,
        'monotonic_time': True,
        'warnings': []
    }
    
    # Merge game outcomes
    df_with_outcomes = df.merge(
        game_outcomes[['GAME_ID', 'home_win']],
        on='GAME_ID',
        how='left'
    )
    
    # Check target label distribution
    logger.info(f"\n1. Target Label Distribution:")
    home_win_counts = df_with_outcomes['home_win'].value_counts(dropna=False)
    logger.info(f"   {home_win_counts.to_dict()}")
    logger.info(f"   NaN count: {df_with_outcomes['home_win'].isna().sum()}")
    
    checks['label_balance'] = {
        'home_win_1': int(home_win_counts.get(1, 0)),
        'home_win_0': int(home_win_counts.get(0, 0)),
        'home_win_nan': int(df_with_outcomes['home_win'].isna().sum())
    }
    
    # Check label consistency per game
    logger.info(f"\n2. Label Consistency Per Game:")
    games_with_outcomes = df_with_outcomes[df_with_outcomes['home_win'].notna()]
    if len(games_with_outcomes) > 0:
        label_per_game = games_with_outcomes.groupby('GAME_ID')['home_win'].nunique()
        inconsistent_games = label_per_game[label_per_game > 1]
        
        if len(inconsistent_games) > 0:
            logger.warning(f"   ✗ Found {len(inconsistent_games)} games with inconsistent home_win labels!")
            logger.warning(f"   Example GAME_IDs: {inconsistent_games.index.tolist()[:5]}")
            checks['label_consistency'] = False
            checks['warnings'].append(f"Found {len(inconsistent_games)} games with inconsistent home_win labels")
        else:
            logger.info(f"   ✓ All games have consistent home_win labels")
    else:
        logger.warning("   ✗ No games with home_win labels found!")
        checks['label_consistency'] = False
        checks['warnings'].append("No games with home_win labels found")
    
    # Check score margin sign symmetry
    logger.info(f"\n3. Score Margin Sign Symmetry:")
    df_with_home_margin = convert_to_home_perspective(df_with_outcomes)
    margin_stats = df_with_home_margin['score_margin_home'].describe()
    logger.info(f"   Min: {margin_stats['min']:.2f}, Max: {margin_stats['max']:.2f}")
    logger.info(f"   Mean: {margin_stats['mean']:.2f}, Std: {margin_stats['std']:.2f}")
    
    positive_count = (df_with_home_margin['score_margin_home'] > 0).sum()
    negative_count = (df_with_home_margin['score_margin_home'] < 0).sum()
    zero_count = (df_with_home_margin['score_margin_home'] == 0).sum()
    
    logger.info(f"   Positive margins: {positive_count} ({positive_count/len(df_with_home_margin)*100:.1f}%)")
    logger.info(f"   Negative margins: {negative_count} ({negative_count/len(df_with_home_margin)*100:.1f}%)")
    logger.info(f"   Zero margins: {zero_count} ({zero_count/len(df_with_home_margin)*100:.1f}%)")
    
    if positive_count == 0 or negative_count == 0:
        logger.warning(f"   ✗ Score margin lacks sign symmetry!")
        checks['margin_sign_symmetry'] = False
        checks['warnings'].append("Score margin lacks sign symmetry (all positive or all negative)")
    else:
        logger.info(f"   ✓ Score margin has both positive and negative values")
    
    # Check monotonic time decrease
    logger.info(f"\n4. Monotonic Time Decrease Check:")
    sample_game_id = df_with_outcomes['GAME_ID'].iloc[0]
    sample_game = df_with_outcomes[df_with_outcomes['GAME_ID'] == sample_game_id].sort_values('EVENTNUM')
    
    if len(sample_game) > 1:
        time_decreasing = all(
            sample_game['total_seconds_remaining'].iloc[i] >= 
            sample_game['total_seconds_remaining'].iloc[i+1]
            for i in range(len(sample_game) - 1)
        )
        
        if not time_decreasing:
            logger.warning(f"   ✗ Time is not monotonically decreasing in sample game {sample_game_id}!")
            logger.warning(f"   Sample: {sample_game[['EVENTNUM', 'total_seconds_remaining']].head(10).to_string()}")
            checks['monotonic_time'] = False
            checks['warnings'].append(f"Time not monotonically decreasing in sample game {sample_game_id}")
        else:
            logger.info(f"   ✓ Time is monotonically decreasing (checked game {sample_game_id})")
    else:
        logger.warning(f"   ✗ Sample game {sample_game_id} has only 1 event, cannot check monotonicity")
        checks['monotonic_time'] = False
    
    # Check for missing values
    logger.info(f"\n5. Missing Values Check:")
    key_cols = ['score_margin_int', 'total_seconds_remaining', 'is_home', 'GAME_ID']
    for col in key_cols:
        if col in df_with_outcomes.columns:
            missing = df_with_outcomes[col].isna().sum()
            if missing > 0:
                logger.warning(f"   ✗ {col}: {missing} missing values ({missing/len(df_with_outcomes)*100:.1f}%)")
                checks['warnings'].append(f"{col} has {missing} missing values")
            else:
                logger.info(f"   ✓ {col}: No missing values")
    
    return checks, df_with_outcomes


def create_visual_diagnostics(df: pd.DataFrame, game_outcomes: pd.DataFrame, output_dir: str):
    """Create visual diagnostic plots."""
    logger.info("=" * 80)
    logger.info("CREATING VISUAL DIAGNOSTICS")
    logger.info("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Merge outcomes and convert to home perspective
    df_with_outcomes = df.merge(
        game_outcomes[['GAME_ID', 'home_win']],
        on='GAME_ID',
        how='left'
    )
    df_with_home_margin = convert_to_home_perspective(df_with_outcomes)
    
    # Filter to rows with valid outcomes
    df_plot = df_with_home_margin[df_with_home_margin['home_win'].notna()].copy()
    
    if len(df_plot) == 0:
        logger.warning("No data with valid outcomes for plotting!")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # (a) Histogram of score_margin_home
    logger.info("Creating histogram of score_margin_home...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_plot['score_margin_home'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Score Margin (Home Team Perspective)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Score Margin (Home Team Perspective)', fontsize=14)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Tie')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'score_margin_hist_{timestamp}.png'), dpi=150)
    plt.close()
    logger.info(f"   Saved: score_margin_hist_{timestamp}.png")
    
    # (b) Histogram of total_seconds_remaining
    logger.info("Creating histogram of total_seconds_remaining...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_plot['total_seconds_remaining'], bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.set_xlabel('Total Seconds Remaining', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Total Seconds Remaining', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'time_remaining_hist_{timestamp}.png'), dpi=150)
    plt.close()
    logger.info(f"   Saved: time_remaining_hist_{timestamp}.png")
    
    # (c) Scatter plot colored by home_win
    logger.info("Creating scatter plot of score_margin_home vs total_seconds_remaining...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sample data for plotting if too large
    plot_df = df_plot.sample(min(10000, len(df_plot))) if len(df_plot) > 10000 else df_plot
    
    scatter = ax.scatter(
        plot_df['total_seconds_remaining'],
        plot_df['score_margin_home'],
        c=plot_df['home_win'],
        cmap='RdYlGn',
        alpha=0.5,
        s=10,
        edgecolors='black',
        linewidth=0.1
    )
    ax.set_xlabel('Total Seconds Remaining', fontsize=12)
    ax.set_ylabel('Score Margin (Home Team Perspective)', fontsize=12)
    ax.set_title('Score Margin vs Time Remaining (colored by home_win)', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Home Win (1=Yes, 0=No)', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scatter_margin_vs_time_{timestamp}.png'), dpi=150)
    plt.close()
    logger.info(f"   Saved: scatter_margin_vs_time_{timestamp}.png")
    
    # (d) WP vs margin curves at fixed times
    logger.info("Creating WP vs margin curves at fixed times...")
    
    # Try to load model predictions if available
    try:
        from models.wp_model import WinProbabilityModel
        wp_model = WinProbabilityModel()
        wp_model.load_model()
        
        # Build features for prediction - use score_margin_home since we've already converted to home perspective
        df_features = build_wp_features(df_plot, score_margin_col='score_margin_home')
        wp_pred = wp_model.predict(df_features)
        df_plot['wp_pred'] = wp_pred
        logger.info("   Loaded existing WP model for predictions")
    except Exception as e:
        logger.warning(f"   Could not load existing model: {e}")
        logger.info("   Will use simple logistic baseline for demonstration")
        df_plot['wp_pred'] = 0.5  # Placeholder
    
    # Create WP vs margin curves at fixed times
    fig, ax = plt.subplots(figsize=(12, 8))
    
    fixed_times = [60, 360, 720]  # 1 min, 6 min, 12 min remaining
    margin_bins = np.arange(-30, 31, 2)
    
    for time_remaining in fixed_times:
        # Filter to events within ±30 seconds of target time
        time_mask = (
            (df_plot['total_seconds_remaining'] >= time_remaining - 30) &
            (df_plot['total_seconds_remaining'] <= time_remaining + 30)
        )
        time_df = df_plot[time_mask].copy()
        
        if len(time_df) > 0:
            # Bin by score margin and compute average WP
            time_df['margin_bin'] = pd.cut(time_df['score_margin_home'], bins=margin_bins)
            wp_by_margin = time_df.groupby('margin_bin', observed=True).agg({
                'wp_pred': 'mean',
                'home_win': 'mean'
            }).reset_index()
            
            # Get bin centers
            wp_by_margin['margin_center'] = wp_by_margin['margin_bin'].apply(
                lambda x: x.mid if pd.notna(x) else np.nan
            )
            wp_by_margin = wp_by_margin[wp_by_margin['margin_center'].notna()]
            
            if len(wp_by_margin) > 0:
                ax.plot(
                    wp_by_margin['margin_center'],
                    wp_by_margin['wp_pred'],
                    'o-',
                    label=f'{time_remaining//60}min remaining (predicted)',
                    linewidth=2,
                    markersize=6
                )
                ax.plot(
                    wp_by_margin['margin_center'],
                    wp_by_margin['home_win'],
                    's--',
                    label=f'{time_remaining//60}min remaining (actual)',
                    linewidth=1,
                    markersize=4,
                    alpha=0.7
                )
    
    ax.set_xlabel('Score Margin (Home Team Perspective)', fontsize=12)
    ax.set_ylabel('Win Probability (Home Team)', fontsize=12)
    ax.set_title('WP vs Score Margin at Fixed Times', fontsize=14)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'wp_vs_margin_curves_{timestamp}.png'), dpi=150)
    plt.close()
    logger.info(f"   Saved: wp_vs_margin_curves_{timestamp}.png")
    
    logger.info(f"\nAll plots saved to {output_dir}")


def run_auc_test(df: pd.DataFrame, game_outcomes: pd.DataFrame) -> Dict:
    """Run AUC test with proper game-level train/test split."""
    logger.info("=" * 80)
    logger.info("RUNNING AUC TEST")
    logger.info("=" * 80)
    
    # Merge outcomes and convert to home perspective
    df_with_outcomes = df.merge(
        game_outcomes[['GAME_ID', 'home_win']],
        on='GAME_ID',
        how='left'
    )
    df_with_home_margin = convert_to_home_perspective(df_with_outcomes)
    
    # Filter to rows with valid outcomes
    df_valid = df_with_home_margin[df_with_home_margin['home_win'].notna()].copy()
    
    if len(df_valid) == 0:
        logger.error("No data with valid outcomes for AUC test!")
        return {'error': 'No valid outcomes'}
    
    # Preserve GAME_ID and home_win before building features
    game_ids = df_valid['GAME_ID'].values
    home_win_labels = df_valid['home_win'].values
    
    # Build features - use score_margin_home since we've already converted to home perspective
    df_features = build_wp_features(df_valid, score_margin_col='score_margin_home')
    
    # Add back GAME_ID and home_win (build_wp_features returns a copy, so index should be preserved)
    df_features['GAME_ID'] = game_ids
    df_features['home_win'] = home_win_labels
    
    # Prepare features for model
    feature_cols = ['score_margin', 'seconds_remaining', 'is_home']
    X = df_features[feature_cols].values
    y = df_features['home_win'].values
    
    # Remove rows with NaN targets
    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) == 0:
        logger.error("No valid data after filtering!")
        return {'error': 'No valid data'}
    
    # Also filter df_features to match valid_mask
    df_features_valid = df_features[valid_mask].copy()
    
    # Split by GAME_ID (70/30 split)
    unique_games = df_features_valid['GAME_ID'].unique()
    game_ids_train, game_ids_test = train_test_split(
        unique_games,
        test_size=0.3,
        random_state=42
    )
    
    train_mask = df_features_valid['GAME_ID'].isin(game_ids_train)
    test_mask = df_features_valid['GAME_ID'].isin(game_ids_test)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    logger.info(f"Train set: {len(X_train)} events from {len(game_ids_train)} games")
    logger.info(f"Test set: {len(X_test)} events from {len(game_ids_test)} games")
    
    # Train logistic regression
    logger.info("Training LogisticRegression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    
    logger.info(f"\nTest Set Metrics:")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives: {cm[0, 0]}")
    logger.info(f"  False Positives: {cm[0, 1]}")
    logger.info(f"  False Negatives: {cm[1, 0]}")
    logger.info(f"  True Positives: {cm[1, 1]}")
    
    # Calibration curve
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        logger.info(f"\nCalibration Summary (10 bins):")
        logger.info(f"  Mean predicted values: {mean_predicted_value}")
        logger.info(f"  Fraction of positives: {fraction_of_positives}")
        
        # Calculate calibration error (mean absolute difference)
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        logger.info(f"  Calibration Error: {calibration_error:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute calibration curve: {e}")
        calibration_error = None
    
    results = {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'brier_score': float(brier),
        'log_loss': float(logloss),
        'confusion_matrix': cm.tolist(),
        'calibration_error': float(calibration_error) if calibration_error is not None else None,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_train_games': len(game_ids_train),
        'n_test_games': len(game_ids_test)
    }
    
    return results


def generate_summary_report(checks: Dict, auc_results: Dict) -> None:
    """Generate and log summary report."""
    logger.info("=" * 80)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 80)
    
    logger.info(f"\nDataset Overview:")
    logger.info(f"  Total rows: {checks['total_rows']:,}")
    logger.info(f"  Unique games: {checks['unique_games']}")
    logger.info(f"  Label balance:")
    logger.info(f"    Home wins (1): {checks['label_balance']['home_win_1']:,}")
    logger.info(f"    Away wins (0): {checks['label_balance']['home_win_0']:,}")
    logger.info(f"    NaN labels: {checks['label_balance']['home_win_nan']:,}")
    
    logger.info(f"\nSanity Verdicts:")
    
    # Label consistency
    verdict = "✓" if checks['label_consistency'] else "✗"
    logger.info(f"  {verdict} Label consistency per game")
    if not checks['label_consistency']:
        logger.warning("    → Some games have inconsistent home_win labels!")
    
    # Margin sign symmetry
    verdict = "✓" if checks['margin_sign_symmetry'] else "✗"
    logger.info(f"  {verdict} Margin sign symmetry")
    if not checks['margin_sign_symmetry']:
        logger.warning("    → Score margin lacks sign symmetry (all positive or all negative)")
    
    # Monotonic time
    verdict = "✓" if checks['monotonic_time'] else "✗"
    logger.info(f"  {verdict} Monotonic time decrease")
    if not checks['monotonic_time']:
        logger.warning("    → Time is not monotonically decreasing in sample game")
    
    # Predictive signal
    if 'auc' in auc_results:
        auc = auc_results['auc']
        verdict = "✓" if auc > 0.7 else "✗"
        logger.info(f"  {verdict} Predictive signal (AUC > 0.7)")
        logger.info(f"    AUC: {auc:.4f}")
        if auc <= 0.7:
            logger.warning(f"    → AUC is {auc:.4f}, below threshold of 0.7")
    else:
        logger.warning("  ✗ Predictive signal (AUC test failed)")
    
    # Warnings summary
    if checks['warnings']:
        logger.info(f"\n⚠️  Warnings ({len(checks['warnings'])}):")
        for i, warning in enumerate(checks['warnings'], 1):
            logger.warning(f"  {i}. {warning}")
    
    # Diagnostic hints if AUC is around 0.5
    if 'auc' in auc_results and auc_results['auc'] < 0.6:
        logger.info(f"\n🔍 DIAGNOSTIC HINTS (AUC = {auc_results['auc']:.4f}):")
        
        if not checks['label_consistency']:
            logger.info("  → LABEL JOIN FAILURE: Check if game_outcomes merge is working correctly")
            logger.info("    - Verify GAME_ID format matches between datasets")
            logger.info("    - Check for missing games in game_outcomes")
        
        if not checks['margin_sign_symmetry']:
            logger.info("  → WRONG MARGIN SIGN: Score margin may not be from home team perspective")
            logger.info("    - Verify score_margin_home conversion logic")
            logger.info("    - Check if is_home column is correctly identified")
        
        if checks['label_balance']['home_win_nan'] > 0:
            logger.info(f"  → MISSING LABELS: {checks['label_balance']['home_win_nan']} rows have NaN labels")
            logger.info("    - Verify game outcome inference logic")
            logger.info("    - Check if all games have valid final score margins")
        
        if auc_results['auc'] < 0.55:
            logger.info("  → DATA LEAKAGE: Model may be learning from future information")
            logger.info("    - Verify time ordering is correct")
            logger.info("    - Check for features that contain future information")
        
        if not checks['margin_sign_symmetry'] or not checks['label_consistency']:
            logger.info("  → FEATURE-TARGET MISMATCH: Features and target may be misaligned")
            logger.info("    - Verify score_margin is from home team perspective")
            logger.info("    - Check if home_win labels match the margin direction")
        
        logger.info("\n💡 RECOMMENDATIONS:")
        logger.info("  1. Verify game_outcomes DataFrame has correct home_win labels")
        logger.info("  2. Check score_margin_home conversion (should flip sign when is_home=0)")
        logger.info("  3. Ensure all games have valid outcomes (no NaN in home_win)")
        logger.info("  4. Verify time ordering is correct (monotonically decreasing)")
        logger.info("  5. Check for data leakage (features should not contain future info)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Diagnostic script for Win Probability model debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/playbyplay_indiana_pacers_2024_25_cleaned.csv',
        help='Path to processed play-by-play CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/triage_wp',
        help='Directory to save diagnostic plots'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("WIN PROBABILITY MODEL DIAGNOSTIC SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load data
    df = load_data(args.data_path)
    
    # Infer game outcomes
    game_outcomes = get_game_outcomes(df)
    
    # Perform integrity checks
    checks, df_with_outcomes = perform_integrity_checks(df, game_outcomes)
    
    # Create visual diagnostics
    create_visual_diagnostics(df, game_outcomes, args.output_dir)
    
    # Run AUC test
    auc_results = run_auc_test(df, game_outcomes)
    
    # Generate summary report
    generate_summary_report(checks, auc_results)
    
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

