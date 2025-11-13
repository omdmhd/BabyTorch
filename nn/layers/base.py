import numpy as np

class BaseLayer:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement forward method")

    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement backward method")

    