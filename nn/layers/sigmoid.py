import numpy as np
from .base import BaseLayer
from nn.tensor import Tensor

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> np.ndarray:
        return x.sigmoid()
    
    def _parameters(self) -> list[Tensor]:
        return []