import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


class MILDataset(Dataset):
    def __init__(self, embedding_dirs: List[str], labels: List[int]) -> None:
        self.embedding_dirs = embedding_dirs
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return np.load(self.embedding_dirs[idx]), self.labels[idx]


def create_mil_datasets(orthopedia_dir: str) -> Dict[str, MILDataset]:
    get_fn = lambda f: pd.read_csv(os.path.join(orthopedia_dir, "csv_files", f))["Fallnummer"].astype(str).to_list() 
    splits = {"train", "val", "test"}
    embedding_dirs = {split: list() for split in splits}
    labels = {split: list() for split in splits}
    fallnummers = {split: get_fn(f"infect_{split}.csv") + get_fn(f"noinfect_{split}.csv") for split in splits}
    for key_folder in ["infect", "noinfect"]:
        key_path = os.path.join(orthopedia_dir, f"{key_folder}_embedding")
        for fn_folder in os.listdir(key_path):
            mapping = [split for split in splits if fn_folder in fallnummers[split]]
            if len(mapping) == 1:
                split = mapping[0]
            else:
                raise ValueError
            fn_path = os.path.join(key_path, fn_folder)
            embedding_dirs[split].extend([os.path.join(fn_path, item) for item in os.listdir(fn_path)])
            labels[split].extend([1 if key_folder == "infect" else 0] * len(os.listdir(fn_path)))
    return {
        split: MILDataset(embedding_dirs[split], labels[split])
        for split in splits
    }


def collate_fn(data: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    features = [torch.from_numpy(item[0]).unsqueeze(0) for item in data]
    labels = torch.tensor([item[1] for item in data], dtype=torch.float)
    return features, labels
