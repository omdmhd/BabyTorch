import numpy as np

class Module:

    def __init__(self):
        self._parameters = []

    def __call__(self, *args, **kwargs):
        # Pre-processing, hooks, etc.
        return self.forward(*args, **kwargs)


    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> list[np.ndarray]:
        return self._parameters