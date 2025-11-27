from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class LSTMWinProbModel(nn.Module):
    """
    LSTM-based Win Probability estimator that consumes sequential play-by-play features.
    Predicts home-team win probability at each time step.
    """

    def __init__(
        self,
        numeric_dim: int,
        vocab_sizes: Dict[str, int],
        embedding_dims: Optional[Dict[str, int]] = None,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        numeric_projection_dim: int = 64,
    ):
        super().__init__()

        default_dims = {
            "event_category": 32,
            "EVENTMSGTYPE": 16,
            "EVENTMSGACTIONTYPE": 16,
            "possession_team": 32,
            "PLAYER1_TEAM_ABBREVIATION": 32,
        }
        if embedding_dims is None:
            embedding_dims = default_dims
        else:
            for key, val in default_dims.items():
                embedding_dims.setdefault(key, val)

        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for name, vocab_size in vocab_sizes.items():
            emb_dim = embedding_dims.get(name, 16)
            self.embeddings[name] = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            total_emb_dim += emb_dim

        self.numeric_proj = nn.Sequential(
            nn.Linear(numeric_dim, numeric_projection_dim),
            nn.ReLU(),
        )

        lstm_input_dim = numeric_projection_dim + total_emb_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(
        self,
        numeric_features: torch.Tensor,
        categorical_features: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            numeric_features: FloatTensor [batch, time, numeric_dim]
            categorical_features: dict of LongTensor [batch, time]
            mask: BoolTensor [batch, time] (optional, unused in forward but kept for API parity)

        Returns:
            logits: FloatTensor [batch, time]
        """

        num_repr = self.numeric_proj(numeric_features)
        embeddings = [
            self.embeddings[name](categorical_features[name]) for name in self.embeddings.keys()
        ]
        lstm_input = torch.cat([num_repr] + embeddings, dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = self.dropout(lstm_out)
        logits = self.output_layer(lstm_out).squeeze(-1)
        return logits


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Binary cross entropy loss that ignores padded timesteps via mask.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    per_step = criterion(logits, targets)
    mask = mask.float()
    loss = (per_step * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return loss


__all__ = ["LSTMWinProbModel", "masked_bce_loss"]

