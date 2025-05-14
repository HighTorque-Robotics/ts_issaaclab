import torch
import torch.nn as nn
from .mlp_encoder import MLPEncoder


class LatentEncoder(MLPEncoder):
    def __init__(self, input_dim, output_dim, hidden_dims, name, activation="elu"):
        super().__init__(input_dim, output_dim, hidden_dims, name, activation="elu")

    def forward(self, x):
        latent = self.network(x)
        return torch.nn.functional.normalize(latent, p=2, dim=-1)
    
    def act_inference(self, x):
        output = self.network(x)
        return torch.nn.functional.normalize(output, p=2, dim=-1)