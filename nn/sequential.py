from neuralnet.module import Module
import numpy as np

class Sequential(Module):
    def __init__(self, layers: list[Module]):
        super().__init__()
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
            print(x.shape)
        return x

    def backward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.backward(x)
        return x