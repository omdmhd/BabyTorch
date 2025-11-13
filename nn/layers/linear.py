from .base import BaseLayer
import torch

class LinearLayer(BaseLayer):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.randn(in_features, out_features, requires_grad=True)
        self.bias = torch.randn(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights + self.bias

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T