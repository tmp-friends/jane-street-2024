import torch
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.1):
        super().__init__()

        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:  # only add noise during training
            noise = torch.randn_like(x) * self.std

            return x + noise

        return x


class LSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_dim: int, output_size: int, num_layers: int
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.noise = GaussianNoise(std=0.1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.noise(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out.squeeze()
