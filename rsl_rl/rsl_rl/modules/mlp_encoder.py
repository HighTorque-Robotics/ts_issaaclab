
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, name, activation="elu"):
        super().__init__()
        self.activation = get_activation(activation)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        print(f"{name} MLP: {self.network}")

    def forward(self, x):
        return self.network(x)
    
    def act_inference(self, x):
        output = self.network(x)
        return output
    

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None