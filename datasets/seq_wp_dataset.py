from __future__ import annotations

from typing import Dict, List, Tuple, Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class WPSequenceDataset(Dataset):
    """
    Dataset wrapper around per-game Win Probability sequences.
    Each item returns numeric features, categorical ID tensors, targets, and masks.
    """

    def __init__(self, game_sequences: Dict[str, Dict[str, Any]]):
        self.game_ids: List[str] = list(game_sequences.keys())
        self.data = game_sequences

    def __len__(self) -> int:
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, Dict[str, torch.LongTensor], torch.FloatTensor, torch.BoolTensor]:
        game_id = self.game_ids[idx]
        entry = self.data[game_id]

        features = torch.as_tensor(entry["features"], dtype=torch.float32)
        target = torch.as_tensor(entry["target"], dtype=torch.float32)
        mask = torch.as_tensor(entry["mask"], dtype=torch.bool)

        categorical = {}
        for key, values in entry.get("categorical", {}).items():
            categorical[key] = torch.as_tensor(values, dtype=torch.long)

        return features, categorical, target, mask


def pad_sequence_collate(
    batch: List[Tuple[torch.FloatTensor, Dict[str, torch.LongTensor], torch.FloatTensor, torch.BoolTensor]]
) -> Tuple[torch.FloatTensor, Dict[str, torch.LongTensor], torch.FloatTensor, torch.BoolTensor]:
    """
    Custom collate_fn for DataLoader to pad variable-length games.
    """

    features_list, categorical_list, targets_list, masks_list = zip(*batch)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=0.0)
    masks_padded = pad_sequence(masks_list, batch_first=True, padding_value=0)

    padded_categoricals: Dict[str, torch.LongTensor] = {}
    if categorical_list and categorical_list[0]:
        cat_keys = categorical_list[0].keys()
        for key in cat_keys:
            seqs = [cat_dict[key] for cat_dict in categorical_list]
            padded_categoricals[key] = pad_sequence(seqs, batch_first=True, padding_value=0)

    return features_padded, padded_categoricals, targets_padded, masks_padded


__all__ = ["WPSequenceDataset", "pad_sequence_collate"]

