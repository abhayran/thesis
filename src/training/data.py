import numpy as np
import os
from PIL import Image
import scipy.io as sio
import torch
from typing import Dict, List, Tuple

from src.training.utils import convert_image_to_torch, convert_seg_map_to_torch, create_tiles, map_array


def parse_lizard_dataset(image_dir: str, label_dir: str, tile_size: int = 128) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    data = {
        key: {"images": list(), "seg_maps": list()}
        for key in {"with_neutrophil", "without_neutrophil"}
    }
    for f in os.listdir(image_dir):
        image = np.array(Image.open(os.path.join(image_dir, f)))
        
        label = sio.loadmat(os.path.join(label_dir, f.replace(".png", ".mat")))
        nuclei_ids, nuclei_classes, inst_map = label["id"].flatten(), label["class"].flatten(), label["inst_map"]
        id_mapping = {nuclei_id: (1 if category == 1 else 0) for nuclei_id, category in zip(nuclei_ids, nuclei_classes)}
        id_mapping[0] = 0
        seg_map = map_array(inst_map, id_mapping)

        image_tiles, seg_map_tiles = create_tiles(image, seg_map, tile_size=tile_size)
        if not len(image_tiles) == len(seg_map_tiles):
            raise Exception(
                f"Number of images: {len(image_tiles)} doesn't match with number of segmentation maps: {len(seg_map_tiles)}"
            )
        for tile_img, tile_seg in zip(image_tiles, seg_map_tiles):
            key = "with_neutrophil" if np.any(tile_seg) else "without_neutrophil" 
            data[key]["images"].append(convert_image_to_torch(tile_img))
            data[key]["seg_maps"].append(convert_seg_map_to_torch(tile_seg))
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images: List[torch.Tensor], seg_maps: List[torch.Tensor]) -> None:
        super().__init__()
        self.images = images
        self.seg_maps = seg_maps
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.seg_maps[idx]
