"""
Build leverage index and contextual weights for Win Probability Added (WPA) calculations.

This module computes leverage weights that emphasize late-game, close-score situations
where individual plays have higher impact on game outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_leverage_weight(
    score_margin: pd.Series,
    seconds_remaining: pd.Series,
    alpha: float = 0.6,
    beta: float = 0.4,
    max_weight: float = 3.0
) -> pd.Series:
    """
    Compute contextual leverage weight for WPA calculations.
    
    Leverage emphasizes:
    - Close games (small absolute score margin)
    - Late game situations (low seconds remaining)
    
    Formula:
        leverage_weight = 1 + α * (1 / (abs(score_margin) + 1)) 
                         + β * (1 / (seconds_remaining / 720 + 1))
    
    Args:
        score_margin: Score margin from home team perspective (can be negative)
        seconds_remaining: Total seconds remaining in game
        alpha: Weight for score margin component (default: 0.6)
        beta: Weight for time remaining component (default: 0.4)
        max_weight: Maximum leverage weight cap (default: 3.0)
        
    Returns:
        Series of leverage weights (clipped between 1.0 and max_weight)
    """
    # Score margin component: closer games → higher weight
    # Add 1 to avoid division by zero
    margin_component = 1 / (np.abs(score_margin) + 1)
    
    # Time remaining component: later in game → higher weight
    # Normalize by 720 seconds (12 minutes = 1 quarter)
    # Add 1 to avoid division by zero
    time_component = 1 / ((seconds_remaining / 720.0) + 1)
    
    # Combine components
    leverage_weight = 1.0 + (alpha * margin_component) + (beta * time_component)
    
    # Clip to reasonable range
    leverage_weight = np.clip(leverage_weight, 1.0, max_weight)
    
    return leverage_weight


def build_wp_features(
    df: pd.DataFrame,
    score_margin_col: str = 'score_margin_int',
    seconds_remaining_col: str = 'total_seconds_remaining',
    is_home_col: str = 'is_home',
    period_col: str = 'PERIOD'
) -> pd.DataFrame:
    """
    Build features for Win Probability model.
    
    Creates feature matrix with:
    - score_margin_int (from home team perspective)
    - total_seconds_remaining
    - is_home (1 if home team, 0 if away)
    - period (categorical, will be one-hot encoded)
    - Optional interaction: score_margin_int / sqrt(total_seconds_remaining + 1)
    
    Args:
        df: DataFrame with play-by-play data
        score_margin_col: Column name for score margin
        seconds_remaining_col: Column name for seconds remaining
        is_home_col: Column name for home/away indicator
        period_col: Column name for period number
        
    Returns:
        DataFrame with feature columns ready for model training
    """
    features_df = df.copy()
    
    # Ensure score margin is from home team perspective
    # If is_home is False, we need to flip the sign (margin is from home perspective)
    # Actually, score_margin_int should already be from home perspective based on our data processing
    # But we'll make sure: if viewing from away team perspective, flip sign
    if score_margin_col in features_df.columns:
        # Score margin is already from home team perspective in our processed data
        features_df['score_margin'] = features_df[score_margin_col]
    else:
        raise ValueError(f"Column '{score_margin_col}' not found in DataFrame")
    
    # Seconds remaining
    if seconds_remaining_col in features_df.columns:
        features_df['seconds_remaining'] = features_df[seconds_remaining_col]
    else:
        raise ValueError(f"Column '{seconds_remaining_col}' not found in DataFrame")
    
    # Home/away indicator (convert boolean to int)
    if is_home_col in features_df.columns:
        features_df['is_home'] = features_df[is_home_col].astype(int)
    else:
        raise ValueError(f"Column '{is_home_col}' not found in DataFrame")
    
    # Period (keep as int for now, will one-hot encode in model)
    if period_col in features_df.columns:
        features_df['period'] = features_df[period_col]
    else:
        raise ValueError(f"Column '{period_col}' not found in DataFrame")
    
    # Optional interaction feature: score_margin / sqrt(seconds_remaining + 1)
    # This captures the interaction between margin and time pressure
    features_df['margin_time_interaction'] = (
        features_df['score_margin'] / 
        np.sqrt(features_df['seconds_remaining'] + 1)
    )
    
    # Replace any infinities or NaNs with 0
    features_df['margin_time_interaction'] = features_df['margin_time_interaction'].replace(
        [np.inf, -np.inf], 0
    ).fillna(0)
    
    return features_df


def get_feature_columns() -> list:
    """
    Get list of feature column names for Win Probability model.
    
    Returns:
        List of feature column names
    """
    return [
        'score_margin',
        'seconds_remaining',
        'is_home',
        'period',
        'margin_time_interaction'
    ]


def validate_features(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate feature DataFrame for completeness and sanity.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Dictionary with validation results
    """
    required_cols = get_feature_columns()
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    validation = {
        'valid': len(missing_cols) == 0,
        'missing_columns': missing_cols,
        'feature_stats': {}
    }
    
    if validation['valid']:
        for col in required_cols:
            validation['feature_stats'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'null_count': int(df[col].isna().sum())
            }
    
    return validation

