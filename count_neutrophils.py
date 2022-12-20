from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import smart_inference_mode

import numpy as np
import os
from PIL import Image
import torch
from typing import List


ORTHOPEDIA_DIR = "/mnt/dss/dssfs04/pn25ke/pn25ke-dss-0001/orthopedia"

CONF_THRESH = 0.25
IOU_THRESH = 0.45
MAX_DET = 1000
MODEL_PATH = "/mnt/dss/dsshome1/09/ge59gew2/thesis/data/best.pt"
CONFIG_PATH = "/mnt/dss/dsshome1/09/ge59gew2/thesis/data/lizard.yaml"
DEVICE_ID = 0
BATCH_SIZE = 256


class Inference:
    def __init__(self, model_path: str, config_path: str, device: torch.device):
        self.model = DetectMultiBackend(model_path, device=device, dnn=False, data=config_path, fp16=False)
        self.device = device
        self.conf_thres = CONF_THRESH
        self.iou_thres = IOU_THRESH
        self.max_det = MAX_DET
    
    @smart_inference_mode()
    def __call__(self, images: List[np.ndarray]):
        x = torch.from_numpy(np.stack(images)).float().to(self.device).permute(0, 3, 1, 2) / 255.
        pred = self.model(x, augment=False, visualize=False)
        return non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)


def make_inference_dir(tile_path: str) -> None:
    path_to_make = tile_path.replace("tiles", "inference")
    if not os.path.exists(path_to_make):
        os.mkdir(path_to_make)


def read_png_into_numpy(png_path: str) -> np.ndarray:
    return np.array(Image.open(png_path))


def main():
    device = torch.device("cpu") if DEVICE_ID < 0 else torch.device(f"cuda:{DEVICE_ID}") 
    inference = Inference(MODEL_PATH, CONFIG_PATH, device)
    for key_folder in ["infect", "noinfect"]:

        key_path = os.path.join(ORTHOPEDIA_DIR, f"{key_folder}_tiles")
        make_inference_dir(key_path)
        
        for fn_folder in os.listdir(key_path):

            fn_path = os.path.join(key_path, fn_folder)
            make_inference_dir(fn_path)

            for img_folder in os.listdir(fn_path):

                print("Running inference on image ID:", img_folder)

                img_path = os.path.join(fn_path, img_folder)                
                img_files = os.listdir(img_path)

                num_iters = len(img_files) // BATCH_SIZE + (len(img_files) % BATCH_SIZE > 0)
                num_neutrophils = list()
                for i in range(num_iters):
                    num_neutrophils.extend([len(preds) for preds in inference([
                        read_png_into_numpy(os.path.join(img_path, f)) for f in img_files[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]])
                    ])
                np.save(os.path.join(fn_path.replace("tiles", "inference"), f"{img_folder}.npy"), np.array(num_neutrophils))


if __name__ == "__main__":
    main()
