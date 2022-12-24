from mil.data import MILDataset, collate_fn
from mil.model import MILLearner
from mil.utils import Logger

import mlflow
import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, List


class Trainer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device("cpu") if config["device_id"] < 0 else torch.device(f"cuda:{config['device_id']}")
        self.model = MILLearner(config["embedding_dim"]).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.loss_fnc = torch.nn.BCELoss()
        # self.logger = Logger(config)

        self.train_dataloader = DataLoader(
            MILDataset(config["orthopedia_dir"]),
            batch_size=config["train_batch_size"],
            shuffle=True,
            collate_fn=collate_fn
        )

    def train_on_batch(self, input_list: List[torch.Tensor], label_tensor: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        pred_tensor = self.model([item.to(self.device) for item in input_list])
        loss = self.loss_fnc(pred_tensor, label_tensor.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        loss = 0. 
        for input_list, label_tensor in self.val_dataloader:
            pred_tensor = self.model([item.to(self.device) for item in input_list])
            loss += self.loss_fnc(pred_tensor, label_tensor.to(self.device))
        self.model.train()
        return loss / len(self.val_dataloader)
    
    def overfit(self) -> None:
        input_list, label_tensor = next(iter(self.train_dataloader))
        with mlflow.start_run():
            for _ in range(self.config["num_epochs"]):
                self.optimizer.zero_grad()
                pred_tensor = self.model([item.to(self.device) for item in input_list])
                loss = self.loss_fnc(pred_tensor, label_tensor.to(self.device))               
                loss.backward()
                self.optimizer.step()
                # self.logger("train_loss", loss.detach().item())
                print(loss.detach().item())
                
    def train(self) -> None:
        with mlflow.start_run():
            for epoch in range(self.config["num_epochs"]):
                epoch_loss = 0.
                for input_list, label_tensor in self.train_dataloader:
                    epoch_loss += self.train_on_batch(input_list, label_tensor)
                self.logger("train_loss", epoch_loss / len(self.train_dataloader))
                self.logger("val_loss", self.validate())
                if (epoch + 1) % self.config["log_model_every"] == 0:
                    mlflow.pytorch.log_model(self.model, f"model_{epoch + 1}")


if __name__ == "__main__":
    import yaml
    with open("mil/config.yaml") as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config)
    trainer.overfit()
