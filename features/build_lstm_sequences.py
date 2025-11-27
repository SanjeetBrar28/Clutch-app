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
    categorical_columns: Tuple[str, ...] = (
        "event_category",
        "EVENTMSGTYPE",
        "EVENTMSGACTIONTYPE",
        "possession_team",
        "PLAYER1_TEAM_ABBREVIATION",
    ),
    vocab_path: Path = VOCAB_PATH,
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
    base_numeric_cols = [
        "score_margin_home",
        "total_seconds_remaining",
        "seconds_remaining_period",
        "abs_score_margin",
        "game_clock_ratio",
        "leverage_index",
        "possession_change",
        "turnover_flag",
        "foul_flag",
    ]
    if numeric_columns:
        base_numeric_cols = numeric_columns

    _ensure_columns(
        df,
        [
            "GAME_ID",
            "EVENTNUM",
            "score_margin_int",
            "tracked_team_is_home",
            "total_seconds_remaining",
            "seconds_remaining_period",
        ],
    )
    # Ensure score_margin_home exists (home-team perspective)
    if "score_margin_home" not in df.columns:
        if "score_margin_int" in df.columns and "tracked_team_is_home" in df.columns:
            df["score_margin_home"] = np.where(
                df["tracked_team_is_home"].astype(int) == 1,
                df["score_margin_int"],
                -df["score_margin_int"],
            )
        else:
            raise ValueError("Need score_margin_int and tracked_team_is_home to compute score_margin_home.")

    # Derived numeric columns
    df["abs_score_margin"] = df["score_margin_home"].abs()
    if "game_clock_ratio" not in df.columns:
        total_seconds = df["total_seconds_remaining"].astype(float)
        df["game_clock_ratio"] = total_seconds / (48 * 60)
    if "leverage_index" not in df.columns:
        df["leverage_index"] = 1.0
    for col in ["possession_change", "turnover_flag", "foul_flag"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0).astype(float)

    df = df.sort_values(["GAME_ID", "EVENTNUM"]).reset_index(drop=True)

    # Numeric normalization stats (z-score)
    # Numeric normalization stats (z-score), excluding binary indicators
    binary_cols = {"possession_change", "turnover_flag", "foul_flag"}
    stats_cols = [col for col in base_numeric_cols if col not in binary_cols]
    numeric_stats = _compute_numeric_stats(df, stats_cols)
    for col in stats_cols:
        mean = numeric_stats[col]["mean"]
        std = numeric_stats[col]["std"]
        df[col] = ((df[col] - mean) / std).fillna(0.0)
    for col in binary_cols:
        numeric_stats[col] = {"mean": 0.0, "std": 1.0}

    # Build categorical vocabularies
    vocab_store: Dict[str, Dict[str, int]] = {}
    categorical_data: Dict[str, np.ndarray] = {}
    for cat_col in categorical_columns:
        if cat_col not in df.columns:
            df[cat_col] = "" if cat_col == "PLAYER1_TEAM_ABBREVIATION" else -1
        if cat_col == "PLAYER1_TEAM_ABBREVIATION":
            df[cat_col] = df[cat_col].fillna("").astype(str).str.upper()
        vocab = _build_vocab(df[cat_col])
        vocab_store[cat_col] = vocab
        categorical_data[cat_col] = _map_series_to_ids(df[cat_col], vocab)

    # Persist vocabularies + normalization stats
    vocab_payload = {
        "categorical": vocab_store,
        "numeric_stats": numeric_stats,
        "numeric_features": base_numeric_cols,
        "categorical_features": list(categorical_columns),
    }
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w") as fp:
        json.dump(vocab_payload, fp, indent=2)

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
        }

    return game_sequences


__all__ = ["build_lstm_sequences", "VOCAB_PATH"]

