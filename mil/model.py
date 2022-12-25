import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Pool(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embedding_dim = config["embedding_dim"]
        self.projection_dim = config["projection_dim"]
        self.num_heads = config["num_attention_heads"]
        self.layer_norm = config["layer_norm"]
        self.query = torch.nn.Parameter(torch.randn(1, 1, self.projection_dim), requires_grad=True)

        self.dropout = config["dropout"]
        self.apply_dropout = self.dropout > 0.

        self.fc_k = nn.Linear(self.embedding_dim, self.projection_dim)
        self.fc_v = nn.Linear(self.embedding_dim, self.projection_dim)
        if self.layer_norm:
            self.ln0 = nn.LayerNorm(self.projection_dim)
            self.ln1 = nn.LayerNorm(self.projection_dim)
        self.fc_o = nn.Linear(self.projection_dim, self.projection_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        K, V = self.fc_k(features), self.fc_v(features)
        if self.apply_dropout:
            K = F.dropout(K, p=self.dropout, training=self.training)
            V = F.dropout(V, p=self.dropout, training=self.training)
        dim_split = self.projection_dim // self.num_heads

        Q_ = torch.cat(self.query.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.projection_dim), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(self.query.size(0), 0), 2)

        if self.layer_norm:
            O = self.ln0(O)
        
        if self.apply_dropout:
            O = O + F.relu(F.dropout(self.fc_o(O), p=self.dropout, training=self.training))
        else:
            O = O + F.relu(self.fc_o(O))
        
        if self.layer_norm:
            O = self.ln1(O)
        return O


class MILLearner(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.pooler = Pool(config)
        self.dropout = config["dropout"]
        self.apply_dropout = self.dropout > 0.
        self.out = nn.Linear(config["projection_dim"], 1)
    
    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        x = torch.stack([self.pooler(item).squeeze() for item in x_list])
        x = F.relu(x)
        if self.apply_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x.squeeze()
