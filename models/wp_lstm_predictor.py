from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from features.build_lstm_sequences import build_lstm_sequences, VOCAB_PATH
from models.wp_seq_model import LSTMWinProbModel


class LSTMWPPredictor:
    """
    Wrapper for loading a trained LSTM win probability model for inference.
    """

    def __init__(
        self,
        model_path: str = "models/artifacts/wp_lstm_model.pt",
        config_path: str = "models/artifacts/wp_lstm_config.json",
        vocab_path: str | Path = VOCAB_PATH,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.vocab_path = Path(vocab_path)
        self.device = torch.device(device)

        if not self.model_path.exists():
            raise FileNotFoundError(f"LSTM model weights not found at {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"LSTM config not found at {self.config_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"LSTM vocab file not found at {self.vocab_path}")

        with self.config_path.open("r") as fp:
            self.config = json.load(fp)
        with self.vocab_path.open("r") as fp:
            self.vocab_payload = json.load(fp)

        self.numeric_features = self.vocab_payload["numeric_features"]
        self.categorical_features = self.vocab_payload["categorical_features"]
        vocab_sizes = {key: len(value) for key, value in self.vocab_payload["categorical"].items()}

        embedding_dims = self.config.get(
            "embedding_dims",
            {"event_category": 32, "EVENTMSGTYPE": 16, "possession_team": 32},
        )

        self.model = LSTMWinProbModel(
            numeric_dim=len(self.numeric_features),
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            hidden_size=self.config.get("hidden_size", 256),
            num_layers=self.config.get("num_layers", 2),
            dropout=self.config.get("dropout", 0.2),
            numeric_projection_dim=self.config.get("numeric_projection_dim", 64),
        )
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_wp(self, df: Any) -> np.ndarray:
        """
        Predict win probability for each event in the provided DataFrame.
        """
        sequences = build_lstm_sequences(
            df,
            numeric_columns=self.numeric_features,
            categorical_columns=tuple(self.categorical_features),
            vocab_path=self.vocab_path,
            use_existing=True,
            vocab_payload=self.vocab_payload,
        )

        predictions = np.zeros(len(df), dtype=np.float32)

        for game_id, seq in sequences.items():
            features = torch.from_numpy(seq["features"]).unsqueeze(0).to(self.device)
            cat_feats = {
                key: torch.from_numpy(values).unsqueeze(0).to(self.device)
                for key, values in seq["categorical"].items()
            }
            mask = torch.from_numpy(seq["mask"]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(features, cat_feats, mask)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            indices = seq.get("indices")
            if indices is None:
                raise ValueError("Sequence is missing original indices for mapping predictions.")
            predictions[indices] = probs

        return predictions

    def evaluate_calibration(self, df: Any, game_outcomes: Any) -> Dict[str, Any]:
        wp_pred = self.predict_wp(df)
        merged = df[['GAME_ID']].copy()
        merged['wp'] = wp_pred
        merged = merged.merge(game_outcomes[['GAME_ID', 'home_win']], on='GAME_ID', how='left')
        y = merged['home_win'].fillna(0).values

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
                    'count': int(mask.sum())
                })
        return {"calibration_data": calibration_data, "bins": bins}

    def plot_calibration(self, df: Any, game_outcomes: Any, save_path: str | None = None):
        import matplotlib.pyplot as plt

        calibration = self.evaluate_calibration(df, game_outcomes)
        pred_rates = [d['predicted_rate'] for d in calibration['calibration_data']]
        actual_rates = [d['actual_rate'] for d in calibration['calibration_data']]

        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(pred_rates, actual_rates, 'o-', label='LSTM calibration')
        plt.xlabel('Predicted Win Probability')
        plt.ylabel('Actual Win Rate')
        plt.title('LSTM Win Probability Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


__all__ = ["LSTMWPPredictor"]

