import numpy as np
from .base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)