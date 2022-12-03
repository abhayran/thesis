import numpy as np
import os
import scipy.io as sio
from PIL import Image

from src.utils import map_array


class LizardDataset:
    def __init__(self, image_dir: str, label_dir: str) -> None:
        self.images, self.seg_maps, self.neutrophil_counts = list(), list(), list()
        for f in os.listdir(image_dir):
            self.images.append(np.array(Image.open(os.path.join(image_dir, f))))
            
            label = sio.loadmat(os.path.join(label_dir, f.replace(".png", ".mat")))
            nuclei_ids, nuclei_classes, inst_map = label["id"].flatten(), label["class"].flatten(), label["inst_map"]
            assert np.allclose(np.unique(inst_map)[1:], nuclei_ids)
            assert len(nuclei_classes) == len(nuclei_ids)
            self.neutrophil_counts.append(np.sum(nuclei_classes == 1))

            id_mapping = {nuclei_id: (1 if category == 1 else 0) for nuclei_id, category in zip(nuclei_ids, nuclei_classes)}
            id_mapping[0] = 0
            self.seg_maps.append(map_array(inst_map, id_mapping))
