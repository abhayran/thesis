import numpy as np
from typing import Dict


def map_array(array: np.ndarray, mapping_dict: Dict[int, int]) -> np.ndarray:
    unique, inverse = np.unique(array, return_inverse=True)
    return np.array([mapping_dict[item] for item in unique])[inverse].reshape(array.shape)
