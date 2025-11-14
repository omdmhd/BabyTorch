from nn.module import Module
from nn.tensor import Tensor

import numpy as np

class Sequential(Module):
    def __init__(self, layers: list[Module]):
        super().__init__()
        self.layers = layers
        self._parameters = []
        for layer in self.layers:
            self._parameters += layer._parameters() 

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    
