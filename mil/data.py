import numpy as np
import os
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class MILDataset(Dataset):
    def __init__(self, orthopedia_dir: str) -> None:
        self.embedding_dirs = list()
        self.labels = list()
        for key_folder in ["infect", "noinfect"]:
            key_path = os.path.join(orthopedia_dir, f"{key_folder}_embedding")
            for fn_folder in os.listdir(key_path):
                fn_path = os.path.join(key_path, fn_folder)
                self.embedding_dirs.extend([os.path.join(fn_path, item) for item in os.listdir(fn_path)])
                self.labels.extend([1 if key_folder == "infect" else 0] * len(os.listdir(fn_path)))
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return np.load(self.embedding_dirs[idx]), self.labels[idx]


def collate_fn(data: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    features = [torch.from_numpy(item[0]).unsqueeze(0) for item in data]
    labels = torch.tensor([item[1] for item in data], dtype=torch.float)
    return features, labels
