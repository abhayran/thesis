from mil.data import create_mil_datasets, collate_fn
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
        self.model = MILLearner(config).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.loss_fnc = torch.nn.BCELoss()
        self.logger = Logger(config)
        
        datasets = create_mil_datasets(config["orthopedia_dir"])
        self.train_dataloader = DataLoader(
            datasets["train"],
            batch_size=config["train_batch_size"],
            shuffle=True,
            collate_fn=collate_fn
        )
        self.val_dataloader = DataLoader(
            datasets["val"],
            batch_size=config["val_batch_size"],
            shuffle=False,
            collate_fn=collate_fn
        )
        # self.test_dataloader = ...

    def train_on_batch(self, input_list: List[torch.Tensor], label_tensor: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        pred_tensor = self.model([item.to(self.device) for item in input_list])
        label_tensor = label_tensor.to(self.device)
        loss = self.loss_fnc(pred_tensor, label_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.detach().item(), torch.eq(pred_tensor > 0.5, label_tensor).sum().item() / len(pred_tensor)
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        loss, acc = 0., 0. 
        for input_list, label_tensor in self.val_dataloader:
            pred_tensor = self.model([item.to(self.device) for item in input_list])
            label_tensor = label_tensor.to(self.device)
            loss += self.loss_fnc(pred_tensor, label_tensor)
            acc += torch.eq(pred_tensor > 0.5, label_tensor).sum().item() / len(pred_tensor)
        self.model.train()
        return loss / len(self.val_dataloader), acc / len(self.val_dataloader)
    
    def overfit(self) -> None:
        input_list, label_tensor = next(iter(self.train_dataloader))
        with mlflow.start_run():
            for _ in range(self.config["num_epochs"]):
                self.optimizer.zero_grad()
                pred_tensor = self.model([item.to(self.device) for item in input_list])
                loss = self.loss_fnc(pred_tensor, label_tensor.to(self.device))               
                loss.backward()
                self.optimizer.step()
                self.logger("train_loss", loss.detach().item())
                
    def train(self) -> None:
        with mlflow.start_run():
            for epoch in range(self.config["num_epochs"]):
                train_epoch_loss, train_epoch_acc = 0., 0.
                for input_list, label_tensor in self.train_dataloader:
                    train_batch_loss, train_batch_acc = self.train_on_batch(input_list, label_tensor)
                    train_epoch_loss += train_batch_loss
                    train_epoch_acc += train_batch_acc
                self.logger("train_loss", train_epoch_loss / len(self.train_dataloader))
                self.logger("train_acc", train_epoch_acc / len(self.train_dataloader))
                val_epoch_loss, val_epoch_acc = self.validate() 
                self.logger("val_loss", val_epoch_loss)
                self.logger("val_acc", val_epoch_acc)
                if (epoch + 1) % self.config["log_model_every"] == 0:
                    mlflow.pytorch.log_model(self.model, f"model_{epoch + 1}")


if __name__ == "__main__":
    import yaml
    with open("mil/config.yaml") as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config)
    trainer.train()
