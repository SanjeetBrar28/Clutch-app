"""
Utilities for converting play-by-play event logs into padded sequences for
training sequential Win Probability models (e.g., LSTMs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

VOCAB_PATH = Path("data/processed/lstm_vocabs.json")


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _build_vocab(values: pd.Series, add_pad: bool = True) -> Dict[str, int]:
    """Create mapping from category value to contiguous IDs."""
    vocab: Dict[str, int] = {}
    next_idx = 0
    if add_pad:
        vocab["<PAD>"] = next_idx
        next_idx += 1
    vocab["<UNK>"] = next_idx
    next_idx += 1

    for item in values.dropna().unique():
        str_item = str(item)
        if str_item not in vocab:
            vocab[str_item] = next_idx
            next_idx += 1
    return vocab


def _map_series_to_ids(series: pd.Series, vocab: Dict[str, int]) -> np.ndarray:
    unk_id = vocab.get("<UNK>", 0)
    return series.fillna("<UNK>").astype(str).map(lambda x: vocab.get(x, unk_id)).to_numpy(dtype=np.int64)


def _compute_numeric_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in columns:
        mean = float(df[col].mean())
        std = float(df[col].std(ddof=0))
        if std == 0 or np.isnan(std):
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def build_lstm_sequences(
    df: pd.DataFrame,
    numeric_columns: List[str] | None = None,
    categorical_columns: Tuple[str, str, str] = ("event_category", "EVENTMSGTYPE", "possession_team"),
    vocab_path: Path = VOCAB_PATH,
    use_existing: bool = False,
    vocab_payload: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Convert play-by-play events into per-game sequences for LSTM training.

    Args:
        df: Processed DataFrame containing league-wide play-by-play.
        numeric_columns: Optional override for numeric feature list.
        categorical_columns: Tuple of categorical column names to encode.
        vocab_path: Where to persist vocabularies + normalization stats.

    Returns:
        Dict mapping GAME_ID to dictionaries containing:
            - features: np.ndarray [T, numeric_dim]
            - target: np.ndarray [T] of home_win labels
            - mask: np.ndarray [T] (ones)
            - categorical: dict of np.ndarray [T] integer IDs
    """

    df = df.copy()
    df["_original_index"] = np.arange(len(df))
    vocab_payload_internal = vocab_payload
    if use_existing:
        if vocab_payload_internal is None:
            if not vocab_path.exists():
                raise FileNotFoundError(f"Vocab file not found at {vocab_path}")
            with vocab_path.open("r") as fp:
                vocab_payload_internal = json.load(fp)
        if vocab_payload_internal is None:
            raise ValueError("Existing vocab payload required for inference.")
        base_numeric_cols = vocab_payload_internal["numeric_features"]
        numeric_stats = vocab_payload_internal["numeric_stats"]
        categorical_columns = tuple(vocab_payload_internal["categorical_features"])
        vocab_store = vocab_payload_internal["categorical"]
    else:
        base_numeric_cols = [
            "score_margin_home",
            "total_seconds_remaining",
            "seconds_remaining_period",
            "home_score",
            "away_score",
            "leverage_index",
            "is_home",
        ]
        if numeric_columns:
            base_numeric_cols = numeric_columns

    _ensure_columns(df, ["GAME_ID", "EVENTNUM"])
    for col in base_numeric_cols:
        if col not in df.columns:
            if col == "score_margin_home":
                if "score_margin_int" in df.columns and "tracked_team_is_home" in df.columns:
                    df["score_margin_home"] = np.where(
                        df["tracked_team_is_home"].astype(int) == 1,
                        df["score_margin_int"],
                        -df["score_margin_int"],
                    )
                else:
                    raise ValueError("Need score_margin_int and tracked_team_is_home to compute score_margin_home.")
            elif col == "leverage_index":
                df["leverage_index"] = 1.0
            else:
                # If column truly missing, create zeros placeholder
                df[col] = 0.0

    df = df.sort_values(["GAME_ID", "EVENTNUM"]).reset_index(drop=True)

    # Numeric normalization stats (z-score)
    if use_existing:
        for col in base_numeric_cols:
            stats = numeric_stats.get(col)
            if stats is None:
                raise ValueError(f"Missing numeric stats for column '{col}' in vocab payload.")
            mean = stats["mean"]
            std = stats["std"]
            if std == 0 or np.isnan(std):
                std = 1.0
            df[col] = ((df[col] - mean) / std).fillna(0.0)
    else:
        numeric_stats = _compute_numeric_stats(df, base_numeric_cols)
        for col in base_numeric_cols:
            mean = numeric_stats[col]["mean"]
            std = numeric_stats[col]["std"]
            df[col] = ((df[col] - mean) / std).fillna(0.0)

    # Build categorical vocabularies
    vocab_store_data: Dict[str, Dict[str, int]] | None = None
    if use_existing:
        vocab_store_data = vocab_payload_internal["categorical"]

    vocab_store: Dict[str, Dict[str, int]] = {}
    categorical_data: Dict[str, np.ndarray] = {}
    if use_existing:
        vocab_store = vocab_store_data or {}
        for cat_col in categorical_columns:
            vocab = vocab_store.get(cat_col)
            if vocab is None:
                raise ValueError(f"Missing vocabulary for categorical column '{cat_col}'.")
            categorical_data[cat_col] = _map_series_to_ids(df[cat_col], vocab)
    else:
        vocab_store: Dict[str, Dict[str, int]] = {}
        for cat_col in categorical_columns:
            if cat_col in df.columns:
                vocab = _build_vocab(df[cat_col])
                vocab_store[cat_col] = vocab
                categorical_data[cat_col] = _map_series_to_ids(df[cat_col], vocab)
            else:
                vocab_store[cat_col] = {"<PAD>": 0, "<UNK>": 1}
                categorical_data[cat_col] = np.full(len(df), 1, dtype=np.int64)

        vocab_payload_internal = {
            "categorical": vocab_store,
            "numeric_stats": numeric_stats,
            "numeric_features": base_numeric_cols,
            "categorical_features": list(categorical_columns),
        }
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with vocab_path.open("w") as fp:
            json.dump(vocab_payload_internal, fp, indent=2)

    # Prepare sequences per game
    game_sequences: Dict[str, Dict[str, Any]] = {}
    def infer_home_win(group: pd.DataFrame) -> float:
        home_view = group[group["tracked_team_is_home"].astype(bool)]
        if not home_view.empty:
            margin = home_view["score_margin_int"].iloc[-1]
        else:
            margin = -group["score_margin_int"].iloc[-1]
        return 1.0 if margin > 0 else 0.0

    home_results = df.groupby("GAME_ID", sort=False).apply(infer_home_win)
    df["home_win"] = df["GAME_ID"].map(home_results).astype(np.float32)

    grouped = df.groupby("GAME_ID", sort=False)
    for game_id, group in grouped:
        group = group.sort_values("EVENTNUM")
        numeric_matrix = group[base_numeric_cols].to_numpy(dtype=np.float32)
        length = numeric_matrix.shape[0]
        target = np.full(length, group["home_win"].iloc[-1], dtype=np.float32)
        mask = np.ones(length, dtype=np.float32)

        cat_seq: Dict[str, np.ndarray] = {}
        for cat_col in categorical_columns:
            cat_seq[cat_col] = categorical_data[cat_col][group.index].astype(np.int64)

        game_sequences[str(game_id)] = {
            "features": numeric_matrix,
            "target": target,
            "mask": mask,
            "categorical": cat_seq,
            "indices": group["_original_index"].to_numpy()
        }

    return game_sequences


__all__ = ["build_lstm_sequences", "VOCAB_PATH"]

