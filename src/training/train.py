import mlflow
import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from src.training.data import parse_lizard_dataset, Dataset
from src.models.unet import UNetTorch
from src.models.loss import dice_loss_torch
from src.training.utils import Logger


class Trainer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        lizard_data = {
            folder: parse_lizard_dataset(
                os.path.join(config["lizard_dir"], "lizard_images", folder), 
                os.path.join(config["lizard_dir"], "lizard_labels", "Labels"), tile_size=config["tile_size"],
            )
            for folder in {"train", "val"}
        }
        self.val_dataloader = DataLoader(
            Dataset(
                lizard_data["val"]["with_neutrophil"]["images"] + lizard_data["val"]["without_neutrophil"]["images"], 
                lizard_data["val"]["with_neutrophil"]["seg_maps"] + lizard_data["val"]["without_neutrophil"]["seg_maps"],
            ),
            batch_size=config["val_batch_size"],
            shuffle=False,
        )
        self.train_dataloaders = {
            key: DataLoader(
                Dataset(
                    lizard_data["train"][key]["images"],
                    lizard_data["train"][key]["seg_maps"],
                ),
                batch_size=config["train_batch_size"],
                shuffle=True,
                num_workers=4,
            )
            for key in {"with_neutrophil", "without_neutrophil"}
        }
        self.train_iterators = {key: iter(value) for key, value in self.train_dataloaders.items()}
        
        device_id = config["device_id"]
        self.device = torch.device("cpu") if device_id < 0 else torch.device(f"cuda:{device_id}")
        self.model = UNetTorch().float().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.logger = Logger(config)

    def sample_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensors, label_tensors = list(), list()
        for key in ["with_neutrophil", "without_neutrophil"]:
            try:
                input_tensor, label_tensor = next(self.train_iterators[key])
            except StopIteration:
                self.train_iterators[key] = iter(self.train_dataloaders[key])
                input_tensor, label_tensor = next(self.train_iterators[key])
            input_tensors.append(input_tensor)
            label_tensors.append(label_tensor)
        return torch.cat(input_tensors, dim=0), torch.cat(label_tensors, dim=0)

    def train_on_batch(self) -> float:
        input_tensor, label_tensor = self.sample_train_data()
        self.optimizer.zero_grad()
        pred_tensor = self.model(input_tensor.to(self.device))
        loss = dice_loss_torch(torch.sigmoid(pred_tensor), label_tensor.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    
    def validate(self) -> float:
        self.model.eval()
        with torch.no_grad():
            loss = 0. 
            for input_tensor, label_tensor in self.val_dataloader:
                pred_tensor = self.model(input_tensor.to(self.device))
                loss += dice_loss_torch(torch.sigmoid(pred_tensor), label_tensor.to(self.device)).item()  
        self.model.train()
        return loss / len(self.val_dataloader)

    def train(self) -> None:
        with mlflow.start_run():
            for iteration in range(self.config["num_iters"]):
                batch_loss = self.train_on_batch()
                self.logger("train_loss", batch_loss)
                if iteration % self.config["validate_every"] == 0:
                    val_loss = self.validate()
                    self.logger("val_loss", val_loss)


if __name__ == "__main__":
    import yaml
    with open(r"src\training\training_config.yaml") as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config)
    trainer.train()
