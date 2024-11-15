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
