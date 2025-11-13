from neuralnet.module import Module
import numpy as np

class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.flatten()

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(-1, 1)