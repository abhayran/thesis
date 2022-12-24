import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Pool(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 4, layer_norm: bool = False) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.query = torch.nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)

        self.fc_k = nn.Linear(embedding_dim, embedding_dim)
        self.fc_v = nn.Linear(embedding_dim, embedding_dim)
        if self.layer_norm:
            self.ln0 = nn.LayerNorm(embedding_dim)
            self.ln1 = nn.LayerNorm(embedding_dim)
        self.fc_o = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        K, V = self.fc_k(features), self.fc_v(features)

        dim_split = self.embedding_dim // self.num_heads

        Q_ = torch.cat(self.query.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.embedding_dim), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(self.query.size(0), 0), 2)

        if self.layer_norm:
            O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        if self.layer_norm:
            O = self.ln1(O)
        return O


class MILLearner(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 4, layer_norm: bool = False) -> None:
        super().__init__()
        self.pooler = Pool(embedding_dim, num_heads, layer_norm)
        self.out = nn.Linear(embedding_dim, 1)
    
    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        x = torch.stack([self.pooler(item).squeeze() for item in x_list])
        x = F.relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x.squeeze()
