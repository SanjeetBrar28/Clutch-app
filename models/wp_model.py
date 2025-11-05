"""
Win Probability (WP) model for estimating team win probability at any point in a game.

This module implements a baseline logistic regression model that predicts
win probability based on game state (score margin, time remaining, etc.).
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from features.build_leverage_features import build_wp_features, get_feature_columns

logger = logging.getLogger(__name__)


class WinProbabilityModel:
    """
    Logistic regression model for predicting win probability.
    """
    
    def __init__(self, model_dir: str = "models/artifacts", random_state: int = 42):
        """
        Initialize Win Probability model.
        
        Args:
            model_dir: Directory to save/load model artifacts
            random_state: Random seed for reproducibility
        """
        self.model_dir = model_dir
        self.random_state = random_state
        self.model = None
        self.encoder = None  # For one-hot encoding period
        self.feature_cols = None
        self.is_trained = False
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"Initialized WinProbabilityModel (model_dir={model_dir})")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare feature matrix X from DataFrame.
        
        Handles one-hot encoding of period and selects numeric features.
        
        Args:
            df: DataFrame with features (from build_wp_features)
            
        Returns:
            Tuple of (feature DataFrame, feature array)
        """
        # Build features if not already built
        if 'score_margin' not in df.columns:
            df = build_wp_features(df)
        
        # One-hot encode period
        period_encoded = pd.get_dummies(df['period'], prefix='period')
        
        # Combine numeric features
        numeric_features = ['score_margin', 'seconds_remaining', 'is_home', 'margin_time_interaction']
        
        # Ensure all numeric features exist
        missing_features = [f for f in numeric_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        X_numeric = df[numeric_features].values
        X_period = period_encoded.values
        
        # Combine features
        X = np.hstack([X_numeric, X_period])
        
        # Store feature column names for later use
        self.feature_cols = numeric_features + list(period_encoded.columns)
        
        # Store encoder info (for later prediction)
        if self.encoder is None:
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            # Fit encoder on period values
            self.encoder.fit(df[['period']])
        
        return df, X
    
    def prepare_target(self, df: pd.DataFrame, game_outcomes: pd.DataFrame) -> np.ndarray:
        """
        Prepare target variable y (home_win = 1 if home team won, else 0).
        
        Args:
            df: DataFrame with play-by-play data
            game_outcomes: DataFrame with columns ['GAME_ID', 'home_win'] (1 if home won, 0 if away won)
            
        Returns:
            Array of target values
        """
        # Merge game outcomes with play-by-play data
        df_with_outcome = df.merge(
            game_outcomes[['GAME_ID', 'home_win']],
            on='GAME_ID',
            how='left'
        )
        
        # Fill missing values (shouldn't happen if all games have outcomes)
        y = df_with_outcome['home_win'].fillna(0).astype(int).values
        
        return y
    
    def train(
        self,
        df: pd.DataFrame,
        game_outcomes: pd.DataFrame,
        test_size: float = 0.2,
        retrain: bool = False
    ) -> Dict[str, float]:
        """
        Train Win Probability model.
        
        Args:
            df: DataFrame with play-by-play data
            game_outcomes: DataFrame with columns ['GAME_ID', 'home_win']
            test_size: Proportion of data to use for testing
            retrain: If True, retrain even if model exists
            
        Returns:
            Dictionary with training metrics (accuracy, AUC, etc.)
        """
        model_path = os.path.join(self.model_dir, 'win_prob_model.pkl')
        
        # Load existing model if available and not retraining
        if not retrain and os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            self.load_model()
            return {'status': 'loaded', 'model_path': model_path}
        
        logger.info("Training new Win Probability model...")
        
        # Prepare features and target
        df_features, X = self.prepare_features(df)
        y = self.prepare_target(df_features, game_outcomes)
        
        # Remove rows with missing target (shouldn't happen, but safety check)
        valid_mask = ~pd.isna(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Training on {len(X)} samples")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Train logistic regression
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'model_path': model_path
        }
        
        logger.info(f"Model trained - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        # Save model
        self.save_model()
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict win probability for home team.
        
        Args:
            df: DataFrame with play-by-play data (must have required features)
            
        Returns:
            Array of win probabilities (0-1 scale)
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        # Prepare features
        df_features, X = self.prepare_features(df)
        
        # Predict probabilities
        wp_proba = self.model.predict_proba(X)[:, 1]  # Probability of home team winning
        
        return wp_proba
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Alias for predict() - returns win probabilities.
        
        Args:
            df: DataFrame with play-by-play data
            
        Returns:
            Array of win probabilities
        """
        return self.predict(df)
    
    def save_model(self, filename: str = 'win_prob_model.pkl'):
        """
        Save trained model to disk.
        
        Args:
            filename: Model filename
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save.")
        
        model_path = os.path.join(self.model_dir, filename)
        
        # Save model and encoder
        model_data = {
            'model': self.model,
            'encoder': self.encoder,
            'feature_cols': self.feature_cols,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename: str = 'win_prob_model.pkl'):
        """
        Load trained model from disk.
        
        Args:
            filename: Model filename
        """
        model_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.encoder = model_data.get('encoder')
        self.feature_cols = model_data.get('feature_cols')
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
    
    def evaluate_calibration(self, df: pd.DataFrame, game_outcomes: pd.DataFrame) -> Dict:
        """
        Evaluate model calibration by plotting predicted vs actual win rates.
        
        Args:
            df: DataFrame with play-by-play data
            game_outcomes: DataFrame with game outcomes
            
        Returns:
            Dictionary with calibration metrics
        """
        # Predict win probabilities
        wp_pred = self.predict(df)
        
        # Get actual outcomes
        y = self.prepare_target(df, game_outcomes)
        
        # Create calibration bins
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(wp_pred, bins)
        
        calibration_data = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() > 0:
                predicted_rate = wp_pred[mask].mean()
                actual_rate = y[mask].mean()
                calibration_data.append({
                    'bin': bins[i-1],
                    'predicted_rate': predicted_rate,
                    'actual_rate': actual_rate,
                    'count': mask.sum()
                })
        
        return {
            'calibration_data': calibration_data,
            'bins': bins
        }
    
    def plot_calibration(self, df: pd.DataFrame, game_outcomes: pd.DataFrame, 
                        save_path: Optional[str] = None):
        """
        Plot calibration curve showing predicted vs actual win rates.
        
        Args:
            df: DataFrame with play-by-play data
            game_outcomes: DataFrame with game outcomes
            save_path: Optional path to save plot
        """
        calibration = self.evaluate_calibration(df, game_outcomes)
        
        pred_rates = [d['predicted_rate'] for d in calibration['calibration_data']]
        actual_rates = [d['actual_rate'] for d in calibration['calibration_data']]
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(pred_rates, actual_rates, 'o-', label='Model calibration')
        plt.xlabel('Predicted Win Probability')
        plt.ylabel('Actual Win Rate')
        plt.title('Win Probability Model Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Calibration plot saved to {save_path}")
        
        plt.close()

