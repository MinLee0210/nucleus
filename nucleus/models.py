import torch.nn as nn

class SubLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, device=device),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class NeuralNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, mid_features: int, n_layers: int, device: str):
        super().__init__()
        self.in_layer = SubLayer(in_features, mid_features, device)
        self.mid_layers = nn.ModuleList([SubLayer(mid_features, mid_features, device) for _ in range(n_layers)])
        self.out_layer = SubLayer(mid_features, out_features, device)

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.out_layer(x)
        return x
