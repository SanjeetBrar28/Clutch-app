#!/usr/bin/env python3
"""
Train an LSTM-based Win Probability model on league-wide play-by-play data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from datasets.seq_wp_dataset import WPSequenceDataset, pad_sequence_collate
from features.build_lstm_sequences import VOCAB_PATH, build_lstm_sequences
from models.wp_seq_model import LSTMWinProbModel, masked_bce_loss


DEFAULT_EMBEDDING_DIMS = {
    "event_category": 32,
    "EVENTMSGTYPE": 16,
    "possession_team": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM Win Probability model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/playbyplay_2025_all_teams_enhanced.csv"),
        help="Path to merged play-by-play CSV.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--numeric-projection-dim", type=int, default=64)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--lr-step-size", type=int, default=5, help="Epoch interval to decay LR (0 disables scheduler).")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Multiplicative factor of LR decay.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_games(
    game_sequences: Dict[str, Dict],
    test_split: float,
    seed: int
) -> Tuple[list, list]:
    from sklearn.model_selection import train_test_split

    game_ids = list(game_sequences.keys())
    labels = [int(game_sequences[gid]["target"][0]) for gid in game_ids]
    train_ids, val_ids = train_test_split(
        game_ids,
        test_size=test_split,
        random_state=seed,
        stratify=labels,
    )
    return train_ids, val_ids


def build_dataloaders(
    sequences: Dict[str, Dict],
    train_ids: list,
    val_ids: list,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = WPSequenceDataset({gid: sequences[gid] for gid in train_ids})
    val_dataset = WPSequenceDataset({gid: sequences[gid] for gid in val_ids})

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_sequence_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_sequence_collate,
    )
    return train_loader, val_loader


def evaluate(
    model: LSTMWinProbModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Optional[float], np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    probs_list = []
    targets_list = []

    with torch.no_grad():
        for features, cat_feats, targets, mask in loader:
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            cat_feats = {k: v.to(device) for k, v in cat_feats.items()}

            logits = model(features, cat_feats, mask)
            loss = masked_bce_loss(logits, targets, mask)
            losses.append(loss.item())

            valid_mask = mask.bool()
            valid_logits = logits[valid_mask]
            valid_targets = targets[valid_mask]
            probs_list.append(torch.sigmoid(valid_logits).cpu().numpy())
            targets_list.append(valid_targets.cpu().numpy())

    if not probs_list:
        return float("nan"), None, np.array([]), np.array([])

    probs = np.concatenate(probs_list)
    y_true = np.concatenate(targets_list)

    auc = None
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, probs))

    return float(np.mean(losses)), auc, probs, y_true


def plot_calibration(probs: np.ndarray, targets: np.ndarray, output_path: Path) -> None:
    if probs.size == 0 or len(np.unique(targets)) < 2:
        return

    frac_pos, mean_pred = calibration_curve(targets, probs, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="LSTM")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("LSTM WP Calibration")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_csv(args.data_path)
    sequences = build_lstm_sequences(df)

    if not sequences:
        raise RuntimeError("No sequences generated from dataset.")

    with VOCAB_PATH.open() as fp:
        vocab_payload = json.load(fp)
    vocab_sizes = {k: len(v) for k, v in vocab_payload["categorical"].items()}
    numeric_dim = next(iter(sequences.values()))["features"].shape[1]

    train_ids, val_ids = split_games(sequences, args.test_split, args.seed)
    train_loader, val_loader = build_dataloaders(
        sequences,
        train_ids,
        val_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = LSTMWinProbModel(
        numeric_dim=numeric_dim,
        vocab_sizes=vocab_sizes,
        embedding_dims=DEFAULT_EMBEDDING_DIMS,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        numeric_projection_dim=args.numeric_projection_dim,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.lr_step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
        )

    history = []
    best_val_auc = -float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, cat_feats, targets, mask in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            cat_feats = {k: v.to(device) for k, v in cat_feats.items()}

            optimizer.zero_grad()
            logits = model(features, cat_feats, mask)
            loss = masked_bce_loss(logits, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_loss, train_auc, _, _ = evaluate(model, train_loader, device)
        val_loss, val_auc, val_probs, val_targets = evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_auc": val_auc,
            }
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_auc={train_auc}, "
            f"val_loss={val_loss:.4f}, val_auc={val_auc}"
        )

        if scheduler:
            scheduler.step()

        improved = val_auc is not None and val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_state = model.state_dict()
            best_epoch = epoch
            plot_calibration(val_probs, val_targets, Path("data/processed/wp_lstm_calibration.png"))
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val_auc={best_val_auc}).")
            break

    artifacts_dir = Path("models/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "wp_lstm_model.pt"
    torch.save(best_state or model.state_dict(), model_path)

    config_data = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "numeric_projection_dim": args.numeric_projection_dim,
        "embedding_dims": DEFAULT_EMBEDDING_DIMS,
        "numeric_features": vocab_payload["numeric_features"],
        "categorical_features": vocab_payload["categorical_features"],
        "model_path": str(model_path),
    }
    config_path = artifacts_dir / "wp_lstm_config.json"
    with config_path.open("w") as fp:
        json.dump(config_data, fp, indent=2)

    metrics_path = Path("data/processed/wp_lstm_training_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as fp:
        json.dump(history, fp, indent=2)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

