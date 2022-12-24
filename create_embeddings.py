import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from torch.autograd import Variable
from typing import List


ORTHOPEDIA_DIR = "/mnt/dss/dssfs04/pn25ke/pn25ke-dss-0001/orthopedia"
INPUT_DIM = 224
EMBEDDING_DIM = 512
DEVICE_ID = 0
BATCH_SIZE = 100


class Embedder:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT).eval().to(self.device)
        
        self.features = {}
        self.model.avgpool.register_forward_hook(self.get_features('feats'))
        
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_DIM, INPUT_DIM)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def get_features(self, name: str):
        def hook(model, input_, output_):
            self.features[name] = output_.detach()
        return hook
        
    @torch.no_grad()
    def __call__(self, image_paths: List[str]):
        input_tensor = Variable(
            torch.stack([
                self.transform(Image.open(path))
                for path in image_paths
            ]).to(self.device)
        )
        self.model(input_tensor)
        return self.features['feats'].squeeze().cpu()


def make_embedding_dir(embedding_path: str) -> None:
    path_to_make = embedding_path.replace("tiles", "embedding")
    if not os.path.exists(path_to_make):
        os.mkdir(path_to_make)


def main():
    device = torch.device("cpu") if DEVICE_ID < 0 else torch.device(f"cuda:{DEVICE_ID}") 
    embedder = Embedder(device)

    for key_folder in ["infect", "noinfect"]:

        key_path = os.path.join(ORTHOPEDIA_DIR, f"{key_folder}_tiles")
        make_embedding_dir(key_path)
        
        for fn_folder in os.listdir(key_path):

            fn_path = os.path.join(key_path, fn_folder)
            make_embedding_dir(fn_path)

            for img_folder in os.listdir(fn_path):

                print("Running inference on image ID:", img_folder)

                img_path = os.path.join(fn_path, img_folder)                
                img_files = os.listdir(img_path)

                num_iters = len(img_files) // BATCH_SIZE + (len(img_files) % BATCH_SIZE > 0)
                embeddings = list()
                for i in range(num_iters):
                    embeddings.append(
                        embedder([
                            os.path.join(img_path, item)
                            for item in img_files[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                        ])
                    )
                np.save(os.path.join(fn_path.replace("tiles", "embedding"), f"{img_folder}.npy"), torch.cat(embeddings).numpy())


if __name__ == "__main__":
    main()
