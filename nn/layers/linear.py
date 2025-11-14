import numpy as np
from .base import BaseLayer
from nn.tensor import Tensor

class LinearLayer(BaseLayer):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor(np.random.randn(in_features, out_features))
        self.bias = Tensor(np.random.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weights + self.bias

    def _parameters(self) -> list[Tensor]:
        return [self.weights, self.bias]