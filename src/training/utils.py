import mlflow
import numpy as np
import torch
from typing import Any, Dict, List, Tuple


def map_array(array: np.ndarray, mapping_dict: Dict[int, int]) -> np.ndarray:
    unique, inverse = np.unique(array, return_inverse=True)
    return np.array([mapping_dict[item] for item in unique])[inverse].reshape(array.shape)


def create_tiles(image: np.ndarray, seg_map: np.ndarray, tile_size: int = 128) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if image.shape[:2] == seg_map.shape:
        (r, c) = seg_map.shape
    else:
        raise ValueError(f"Image shape: {image.shape} does not match segmentation mask shape: {seg_map.shape}")
    get_num_tiles = lambda x: x // tile_size + (x % tile_size != 0)
    num_tiles_r, num_tiles_c = get_num_tiles(r), get_num_tiles(c)
    image_tiles, seg_map_tiles = list(), list()
    for i in range(num_tiles_r):
        for j in range(num_tiles_c):
            r_end, c_end = min((i + 1) * tile_size, r), min((j + 1) * tile_size, c)
            image_tiles.append(image[r_end - tile_size : r_end, c_end - tile_size : c_end, :])
            seg_map_tiles.append(seg_map[r_end - tile_size : r_end, c_end - tile_size : c_end])
    return image_tiles, seg_map_tiles


def convert_image_to_torch(image: np.ndarray) -> torch.Tensor:
    return torch.Tensor(image).float().permute(2, 0, 1) / 255.0


def convert_seg_map_to_torch(seg_map: np.ndarray) -> torch.Tensor:
    return torch.Tensor(seg_map).float().unsqueeze(0) / 255.0


class Logger:
    def __init__(self, config: Dict) -> None:
        self.history = dict()
        mlflow.set_tracking_uri(config["mlflow_uri"])
        mlflow.set_experiment(experiment_name=config["mlflow_experiment_name"])

    def __call__(self, key: str, value: Any) -> None:
        if key in self.history:
            self.history[key] += 1
        else:
            self.history[key] = 0
        mlflow.log_metric(key=key, value=value, step=self.history[key])
